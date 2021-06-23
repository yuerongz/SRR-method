from osgeo import ogr
import numpy as np
import os, math
import scipy.spatial.distance as dist


def directory_checking(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print("Directory ", target_dir, " Created.")
    else:
        print("Directory ", target_dir, " already exists.")


def dem_checking(demtif):
    with open(demtif) as f:
        print('DEM in .tif format is available~~ Thanks!')
    return demtif


def aggregate_arr(arr, transform, factor, agg_func=np.nanmean):
    wlarr = arr.copy()
    resample_shape = (wlarr.shape[0] // factor, wlarr.shape[1] // factor)
    wlarr = wlarr[0:resample_shape[0] * factor, 0:resample_shape[1] * factor]
    sh = resample_shape[0], factor, resample_shape[1], factor
    wlarr_resample = agg_func(agg_func(wlarr.reshape(sh), axis=-1), axis=1)
    wltransform = transform
    resample_transform = (wltransform[0], wltransform[1] * factor, wltransform[2],
                          wltransform[3], wltransform[4], wltransform[5] * factor)
    return wlarr_resample, resample_transform


def points_along_line(line_ls, distance, start_offset=0, end_offset=0):
    length_by_pt = np.zeros(len(line_ls))
    for i in range(len(line_ls)-1):
        curr_distance = np.sqrt(((line_ls[i+1][0]-line_ls[i][0])**2) + ((line_ls[i+1][1]-line_ls[i][1])**2))
        length_by_pt[i+1] = curr_distance + length_by_pt[i]
    interpolating_dists = np.arange(start_offset, length_by_pt[-1]-end_offset, distance)
    xs = np.interp(interpolating_dists, length_by_pt, [coords[0] for coords in line_ls])
    ys = np.interp(interpolating_dists, length_by_pt, [coords[1] for coords in line_ls])
    return [(xs[i], ys[i]) for i in range(xs.shape[0])]


def perform_DUPLEX_to_pts_for_sdr_rl(potential_pts, ratio_of_choose):
    # choose points based on coordinates by DUPLEX method
    no_of_chosen_pts = math.ceil(len(potential_pts) * ratio_of_choose)
    potential_pts = np.array(potential_pts)
    starting_pts = np.array([])
    all_pts = potential_pts
    dist_m = dist.cdist(potential_pts, potential_pts, 'sqeuclidean')
    r1, r2 = np.argwhere(dist_m == dist_m.max())[0]
    dist_m[dist_m == 0] = np.nan
    chosen_pts_idxs = -np.ones(no_of_chosen_pts, dtype=int)
    chosen_pts_idxs[[0, 1]] = [r1, r2]
    remaining_idxs = np.array(range(potential_pts.shape[0]))
    remaining_idxs[[r1, r2]] = -1
    for i in range(no_of_chosen_pts-2):
        r_of_remain = np.nanargmax(np.nanmin(
            dist_m[:, chosen_pts_idxs[:i + 2]][remaining_idxs[remaining_idxs != -1]], axis=1))
        target_idx = remaining_idxs[remaining_idxs != -1][r_of_remain]
        chosen_pts_idxs[i + 2] = target_idx
        remaining_idxs[target_idx] = -1
    chosen_pts = all_pts[chosen_pts_idxs[starting_pts.shape[0]:], :]
    chosen_pts = [tuple(item) for item in chosen_pts]
    return chosen_pts


def save_to_shp_points_for_sdr_rl(pts, outfile):
    shpDriver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outfile):
        shpDriver.DeleteDataSource(outfile)
    outDataSource = shpDriver.CreateDataSource(outfile)
    outLayer = outDataSource.CreateLayer(outfile, geom_type=ogr.wkbPoint)
    featureDefn = outLayer.GetLayerDefn()
    for coords in pts:
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(coords[0], coords[1])
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(point)
        outLayer.CreateFeature(outFeature)
        del point, outFeature

