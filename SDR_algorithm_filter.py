from gdal_func import coords2rc, ogr, gdal
import numpy as np
import fiona, os, math
from SDR_algorithm_functions import dem_checking, directory_checking, points_along_line
from SDR_algorithm_functions import perform_DUPLEX_to_pts_for_sdr_rl, save_to_shp_points_for_sdr_rl
from time import time
import scipy.spatial.distance as dist


class Select_Rep_pts:
    """
    Select representative points from the trajectories saved in SHP file
    """

    def __init__(self, result_work_dir, demfile, traj_shp, rep_traj_ratio=1 / 200, stoping_categories=(-555, -666)):
        self.work_dir = result_work_dir
        self.dem_dataset = gdal.Open(demfile)
        self.dem_transform = self.dem_dataset.GetGeoTransform()
        self.demarr = self.dem_dataset.GetRasterBand(1).ReadAsArray()
        self.drainage_map = np.zeros(self.demarr.shape)
        self.stopping_vals = stoping_categories
        self.countline = [0]
        self.trajfile = traj_shp
        self.representative_trajs_ratio = rep_traj_ratio

    def read_shp_to_trajs(self, shpfile, need_failed_trajs=False):
        trajs = dict()
        failed_pts = list()
        with fiona.open(shpfile) as copy_shp:
            for feature in copy_shp:
                geom = feature['geometry']['coordinates']
                trajs[geom[0]] = geom[1:]
                r, c = coords2rc(self.dem_transform, trajs[geom[0]][-1])
                if self.is_not_stopping_grids(self.demarr[r, c]):
                    failed_pts.append(geom[0])  # [(x,y), (x,y), ...]
        self.countline[0] = len(list(trajs.keys())) - len(failed_pts)
        print(f'The total number of initially successful trajectories: {self.countline[0]}/{len(list(trajs.keys()))}')
        if need_failed_trajs:
            return trajs, failed_pts
        else:
            return trajs

    def is_not_stopping_grids(self, grid_val):
        if np.isnan(grid_val):
            return False
        for stop_val in self.stopping_vals:
            if grid_val == stop_val:
                return False
        return True

    def save_to_shp_lines(self, trajs, outfile):
        shpDriver = ogr.GetDriverByName("ESRI Shapefile")
        if os.path.exists(outfile):
            shpDriver.DeleteDataSource(outfile)
        outDataSource = shpDriver.CreateDataSource(outfile)
        outLayer = outDataSource.CreateLayer(outfile, geom_type=ogr.wkbMultiLineString)
        featureDefn = outLayer.GetLayerDefn()
        for key in trajs.keys():
            multiline = ogr.Geometry(ogr.wkbMultiLineString)
            line = ogr.Geometry(ogr.wkbLineString)
            line.AddPoint(key[0], key[1])
            for pts in trajs[key]:
                line.AddPoint(pts[0], pts[1])
            multiline.AddGeometry(line)
            outFeature = ogr.Feature(featureDefn)
            outFeature.SetGeometry(multiline)
            outLayer.CreateFeature(outFeature)
            del multiline, line, outFeature

    def save_to_shp_points(self, pts, outfile):
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

        # trajectory filter

    def trajs_filtering(self, trajs):
        """
        output:
            traj_categories = {cate_val0:traj_cate_0, cate_val1:traj_cate_1}
                traj_cates = {coords_of_last_pt_of_sig_traj:traj_group_0, }
                    traj_group: dict((st_coords):[line_ls], ...,
                                    'sig_traj':[line_ls], ...)
        """
        traj_categories = dict()
        for category_val in self.stopping_vals:
            traj_categories[category_val] = dict()
        for st_coords in trajs.keys():
            if len(trajs[st_coords]) <= 20:  # delete trajs with less than 20 points
                continue
            end_val = self.demarr[coords2rc(self.dem_transform, trajs[st_coords][-1])]
            if np.isnan(end_val):  # delete trajs with nan at end
                continue
            else:
                for category_val in self.stopping_vals:
                    if end_val == category_val:  # fall into one category (-555 / -666)
                        if trajs[st_coords][-1] in traj_categories[category_val].keys():
                            traj_categories[category_val][trajs[st_coords][-1]][st_coords] = trajs[st_coords]
                        # elif min_dist_met:  # the distance between the current last point and an existing key pt is less than a threshold ?
                        #     # append ? swap ? need ?
                        else:
                            traj_categories[category_val][trajs[st_coords][-1]] = dict()
                            traj_categories[category_val][trajs[st_coords][-1]][st_coords] = trajs[st_coords]
                        break
        # summarizing each group in main category (ending in the river)
        main_category = self.stopping_vals[0]  # -555, river
        main_cate_dict = traj_categories[main_category]
        traj_categories[main_category] = self.trajs_filtering_summary_main_cate(main_cate_dict)
        return traj_categories

    def trajs_filtering_summary_main_cate(self, main_cate_dict):
        for last_pt_coords in main_cate_dict.keys():  # each group; ending with the same grid
            # summarise significant trajs, their length, coverage extent/area, near river dem
            group_dict = main_cate_dict[last_pt_coords]
            length_ls = [len(line_ls) for line_ls in list(group_dict.values())]
            longest_idx = length_ls.index(max(length_ls))
            # choose representative starting pts from each 'group' with a chosing ratio
            potential_pts = list(group_dict.keys())
            potential_pts.remove(list(group_dict.keys())[longest_idx])
            if len(potential_pts) - 2 > 0:
                choosed_pts = [list(group_dict.keys())[longest_idx], last_pt_coords]
                main_cate_dict[last_pt_coords]['rep_starting_pts'] = self.choose_pts_DUPLEX(potential_pts,
                                                                                            choosed_pts,
                                                                                            self.representative_trajs_ratio)
                main_cate_dict[last_pt_coords]['rep_starting_pts'].append(list(group_dict.keys())[longest_idx])
            else:
                main_cate_dict[last_pt_coords]['rep_starting_pts'] = [list(group_dict.keys())[longest_idx]]
            main_cate_dict[last_pt_coords]['significant_traj'] = group_dict[list(group_dict.keys())[longest_idx]]
            main_cate_dict[last_pt_coords]['near_river_dem'] = self.demarr[coords2rc(
                self.dem_transform, main_cate_dict[last_pt_coords]['significant_traj'][-2])]
        return main_cate_dict

    def choose_pts_DUPLEX(self, potential_pts, starting_pts, ratio_of_choose):
        # choose points based on coordinates by DUPLEX method
        no_of_chosen_pts = math.ceil(len(potential_pts) * ratio_of_choose)
        potential_pts = np.array(potential_pts)
        if starting_pts is None:
            starting_pts = np.array([])
            all_pts = potential_pts
            dist_m = dist.cdist(potential_pts, potential_pts, 'sqeuclidean')
            r1, r2 = np.argwhere(dist_m == dist_m.max())[0]
            dist_m[dist_m == 0] = np.nan
            chosen_pts_idxs = -np.ones(no_of_chosen_pts, dtype=int)
            chosen_pts_idxs[[0, 1]] = [r1, r2]
            remaining_idxs = np.array(range(potential_pts.shape[0]))
            remaining_idxs[[r1, r2]] = -1
            for i in range(no_of_chosen_pts - 2):
                r_of_remain = np.nanargmax(np.nanmin(
                    dist_m[:, chosen_pts_idxs[:i + 2]][remaining_idxs[remaining_idxs != -1]], axis=1))
                target_idx = remaining_idxs[remaining_idxs != -1][r_of_remain]
                chosen_pts_idxs[i + 2] = target_idx
                remaining_idxs[target_idx] = -1
        else:
            starting_pts = np.array(starting_pts)
            all_pts = np.concatenate((starting_pts, potential_pts), axis=0)
            dist_m = dist.cdist(all_pts, all_pts, 'sqeuclidean')
            dist_m[dist_m == 0] = np.nan
            chosen_pts_idxs = -np.ones(no_of_chosen_pts, dtype=int)
            chosen_pts_idxs = np.concatenate((np.array(range(starting_pts.shape[0])), chosen_pts_idxs), axis=0)
            remaining_idxs = np.array(range(all_pts.shape[0]))
            remaining_idxs[np.array(range(starting_pts.shape[0]))] = -1
            for i in range(no_of_chosen_pts):
                r_of_remain = np.nanargmax(np.nanmin(
                    dist_m[:, chosen_pts_idxs[:i + starting_pts.shape[0]]][remaining_idxs[remaining_idxs != -1]],
                    axis=1))
                target_idx = remaining_idxs[remaining_idxs != -1][r_of_remain]
                chosen_pts_idxs[i + starting_pts.shape[0]] = target_idx
                remaining_idxs[target_idx] = -1
        chosen_pts = all_pts[chosen_pts_idxs[starting_pts.shape[0]:], :]
        chosen_pts = [tuple(item) for item in chosen_pts]
        return chosen_pts

    def extract_rep_trajs(self, traj_category):
        rep_trajs = {}
        main_cate_dict = traj_category[self.stopping_vals[0]]
        for last_pt_coords in main_cate_dict.keys():
            key_ls = main_cate_dict[last_pt_coords]['rep_starting_pts']
            for ky in key_ls:
                rep_trajs[ky] = main_cate_dict[last_pt_coords][ky]
        return rep_trajs

    def save_rep_trajs_2shp_with_this_ratio(self, outshp, filter_ratio):
        self.representative_trajs_ratio = filter_ratio
        trajs = self.read_shp_to_trajs(self.trajfile)
        filtered_trajs_category = self.trajs_filtering(trajs)
        self.save_to_shp_lines(self.extract_rep_trajs(filtered_trajs_category), outshp)
        return 0


def extract_rep_pts(mcl_file, sdr_thalwegs_file, resample_rate_mcl, resample_rate_sdr_thalwegs,
                    results_dir, outshp):
    trajs = dict()
    with fiona.open(sdr_thalwegs_file) as copy_shp:
        for feature in copy_shp:
            geom = feature['geometry']['coordinates']
            trajs[geom[0]] = geom[1:]
    last_pt_ls = list(set([trajs[c_k][-1] for c_k in trajs.keys()]))
    trajs = dict()
    with fiona.open(mcl_file) as copy_shp:
        for feature in copy_shp:
            geom = feature['geometry']['coordinates']
            trajs[geom[0]] = geom[1:]
    river_line_ls = list(trajs.values())[0]
    river_pts = points_along_line(river_line_ls, resample_rate_mcl, resample_rate_mcl / 2, resample_rate_mcl / 2)
    selected_pts = perform_DUPLEX_to_pts_for_sdr_rl(last_pt_ls, resample_rate_sdr_thalwegs) + river_pts
    save_to_shp_points_for_sdr_rl(selected_pts, f"{results_dir}{outshp}")
    print(f"Selected totally {len(selected_pts)} RLs, with {len(river_pts)} along MCL")
