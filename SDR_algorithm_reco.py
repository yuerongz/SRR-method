from gdal_func import gdal_asarray, gdal_writetiff, gdal_transform, rc2coords, read_shp_point
from scipy.interpolate import griddata
import csv, fiona
import numpy as np


def read_csv(csvfile_name):
    output = []
    with open(csvfile_name, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            output.append(row)
    return output


def inundation_compare(rebuild_map_arr, ref_map_arr, outfile, ref_file):
    inundation_rebuild = rebuild_map_arr.copy()
    inundation_rebuild[np.isnan(inundation_rebuild)] = -999
    inundation_ref = ref_map_arr.copy()
    inundation_ref[np.isnan(inundation_ref)] = -999
    compare_map = inundation_ref[:,:]
    compare_map = inundation_rebuild - compare_map
    compare_map[(inundation_rebuild == -999) & (inundation_ref == -999)] = np.nan
    compare_map[(inundation_rebuild != -999) & (inundation_ref == -999)] = -888   # false alarm
    compare_map[(inundation_rebuild == -999) & (inundation_ref != -999)] = -777   # miss
    gdal_writetiff(compare_map, outfile, ref_file)
    return 0


def read_lines_to_coordinates_bank(shpfile):
    coords_bank = dict()  # use the end point as key, append all points of trajs ending at that RL
    trajs = dict()
    with fiona.open(shpfile) as copy_shp:
        for feature in copy_shp:
            geom = feature['geometry']['coordinates']
            trajs[geom[0]] = geom[1:]
            if geom[-1] in coords_bank.keys():
                coords_bank[geom[-1]].extend(geom[:-1])
            else:
                coords_bank[geom[-1]] = geom
    return coords_bank, trajs


def get_tidal_coords_bank(tidal_boundary_file):
    transform = gdal_transform(tidal_boundary_file)
    tidal_boundary_arr = gdal_asarray(tidal_boundary_file)
    tidal_boundary_arr[np.isnan(tidal_boundary_arr)] = 0    # value=1 is tidal bournday grids
    rs, cs = np.where(tidal_boundary_arr == 1)
    tidal_coords = [rc2coords(transform, (rs[i], cs[i])) for i in range(len(rs))]
    return tidal_coords


def dem_cut(demfile, grd_surface):
    demarr = gdal_asarray(demfile)
    inundation_map = grd_surface[:,:]
    inundation_map[np.isnan(inundation_map)] = -999
    demarr[np.isnan(demarr)] = -999
    inundation_map[demarr == -999] = -999
    inundation_map[inundation_map < demarr] = -999
    inundation_map[inundation_map == -999] = np.nan
    return inundation_map


def simple_rls_surface_rebuild(rls_coords, coords_bank, wls, target_xy_grids,
                               tidal_coords_bank=None, sealvl=None, if_cut_dem=False, demfile=None):
    points = []
    water_levels = []
    # include only drainage paths ends at selected RLs
    for rl_idx in range(len(rls_coords)):
        rl = rls_coords[rl_idx]
        if rl in coords_bank.keys():
            points.extend(coords_bank[rl])
            water_levels.extend([wls[rl_idx]]*len(coords_bank[rl]))
        else:
            points.append(rl)
            water_levels.append(wls[rl_idx])
    # add sea level into fitting process
    if tidal_coords_bank is not None:
        if sealvl is None:
            raise TypeError('No water level information is provided for downstream conditions')
        for sea_pt in tidal_coords_bank:
            points.append(sea_pt)
            water_levels.append(sealvl)
    grd_surface = griddata(np.array(points), np.array(water_levels), target_xy_grids, method='linear')
    if if_cut_dem:
        if demfile is None:
            raise TypeError('DEM file is not provided. It needs to be in .tif format and indicate boundary grids with value=1.')
        inundation_map = dem_cut(demfile, grd_surface)
        return inundation_map
    else:
        return grd_surface


def two_step_surface_rebuild(rls_coords, coords_bank, wls, x_coords, y_coords ,
                             tidal_coords_bank=None, sealvl=None, demfile=None):
    full_rls_coords = rls_coords.copy()
    full_wls = wls.copy()
    # build step 1 inundation map, and extract water levels at assistant RLs (those not in RLs list)
    arl_ls = [arl for arl in coords_bank.keys() if arl not in rls_coords]
    inundation_map_step_1 = simple_rls_surface_rebuild(rls_coords, coords_bank, wls, np.array(arl_ls),
                                                       tidal_coords_bank, sealvl)
    # include additional RLs not predicted by DLs
    full_rls_coords.extend(arl_ls)
    full_wls.extend(inundation_map_step_1)
    # build for model domain
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    inundation_map_step_2 = simple_rls_surface_rebuild(full_rls_coords, coords_bank, full_wls, (grid_x, grid_y),
                                                       tidal_coords_bank, sealvl, if_cut_dem=True, demfile=demfile)
    return inundation_map_step_2


def reconstruct_flood_inundation_map(demfile, targeted_x_coords, targeted_y_coords,
									 rl_shp_file, sdr_thalwegs_shp_file, water_levels_pred,
									 tidal_boundary_file=None, sealvl=None,
									 save_to_tif=None, reference_tif_file=None):
    """
    Reconstruct flood inundation map according to the below provided information
    :param demfile: DEM file in GeoTiff(.tif) format
    :param targeted_x_coords: 1D numpy array of the x coordinates where the water levels need to be constructed
    :param targeted_y_coords: 1D numpy array of the y coordinates where the water levels need to be constructed
    :param rl_shp_file: the SDR-RL output shapefile, containing RL points
    :param sdr_thalwegs_shp_file: the SDR-Searching output shapefile, containing filtered SDR thalwegs
    :param water_levels_pred: a list of water levels corresponding to each RL in the order of FID in rl_shp_file
    :param tidal_boundary_file: a raster at the same resolution as DEM file,
                                where the downstream boundary grids were marked with value=1,
                                if provided, downstream water level will be used in reconstruction
    :param sealvl: the downstream water level
    :return: 2D numpy array, or Geotiff file if required
    """
    coords_bank, _ = read_lines_to_coordinates_bank(sdr_thalwegs_shp_file)
    rls_coords = read_shp_point(rl_shp_file)
    # retrieve sea level information if provided
    if tidal_boundary_file is not None:
        tidal_coords_bank = get_tidal_coords_bank(tidal_boundary_file)
    else:
        print('Note: No downstream boundary water level was provided, proceeding without it.')
        tidal_coords_bank = None
    inundation_map_pred = two_step_surface_rebuild(rls_coords, coords_bank, water_levels_pred,
                                                   targeted_x_coords, targeted_y_coords,
                                                   tidal_coords_bank, sealvl, demfile)
    if save_to_tif is None:
        return inundation_map_pred
    else:
        gdal_writetiff(inundation_map_pred, save_to_tif, reference_tif_file)

