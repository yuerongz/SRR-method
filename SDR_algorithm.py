from gdal_func import coords2rc, rc2coords, ogr, gdal
import numpy as np
import fiona, os
from time import time
from SDR_algorithm_fetch_startpoints import get_starting_pts_from_maximum_inundation_extent
from SDR_algorithm_functions import dem_checking, directory_checking, aggregate_arr


class Traj_search:
    """
	Class inputs:
		result_work_dir: the directory where the results will be saved
		starting_pts: numpy array of shape (N, 2), N number of coordinates (x, y)
		demfile: a string indicating the DEM file (accept only .tif format)
		stopping_categories: default=(-555, -666)
		search_win_size: default=9
		cluster_threshold: default=1, used for unclustering trajectories
		
    General notes:
		DEM grid values representations during searching:
			default stopping_categories:
				-555: Non-flood mainstream water body area
				-666: Downstream boundary grids
				Note: the first category will be used as main category for trajectory classification!
			np.nan: outside of model/search domain
			np.inf: searched grids for the current trajectory
			555: temporarily exist, for indicating the current grid when decide next point

		drainage_map:
			Has the same shape/size as the DEM array, indicating grids of successfully found trajectories
			Values are Integers:
				0: not marked in all successful trajectories
				1-x: the index of successful trajectories plus 1;
					 when used to choose trajectory, use trajs[success_pts[x-1]]
    """
    def __init__(self, result_work_dir, starting_pts, demfile, stoping_categories=(-555, -666), search_win_size=9,
                 cluster_threshold=1):
        self.work_dir = result_work_dir
        self.starting_pts = starting_pts
        self.dem_dataset = gdal.Open(demfile)
        self.dem_transform = self.dem_dataset.GetGeoTransform()
        self.demarr = self.dem_dataset.GetRasterBand(1).ReadAsArray()
        self.countline = [0]
        self.drainage_map = np.zeros(self.demarr.shape)
        self.default_search_win_size = search_win_size
        self.stopping_vals = stoping_categories
        self.straighten_degree = cluster_threshold

        # standardise starting coordinates
        for pt_idx in range(len(starting_pts)):
            self.starting_pts[pt_idx, :] = rc2coords(self.dem_transform,
                                                     coords2rc(self.dem_transform, starting_pts[pt_idx, :]))
        del pt_idx

    def initial_stpts_trajs(self, search_window_size=9, save_to_shp=False, outfilename=None):
        assert search_window_size % 3 == 0, 'The searching window size must be a multiple of 3!'
        trajs = self.construct_lines_map(self.starting_pts, search_window_size)
        if save_to_shp:
            if outfilename is None:
                outfilename = f"{self.work_dir}trajs_initial_search_from_starting_points.shp"
            self.save_to_shp_lines(trajs, outfilename)
        print(f'The total number of initially successful trajectories: {self.countline[0]}/{len(list(trajs.keys()))}')
        return trajs

    def construct_lines_map(self, stpts, search_win_size):
        traj_collection = dict()
        for i in range(stpts.shape[0]):
            # print(f"traj_id: {i}")
            traj_collection[(stpts[i, 0], stpts[i, 1])] = self.single_line_search(stpts[i, :], search_win_size)
        return traj_collection

    def single_line_search(self, st_coords, search_win_size, modified_dem=None):
        if modified_dem is None:
            demarr_cl = self.demarr.copy()
        else:
            demarr_cl = modified_dem.copy()
        line_ls = [tuple(st_coords)]   # store pts along line
        r, c = coords2rc(self.dem_transform, st_coords)
        demarr_cl[r, c] = 555    # mark current point as 555
        # define terminate criteria test window boundaries, always 3x3
        up0, dwn0, lf0, rit0 = max(0, r - 1), min(demarr_cl.shape[0], r + 1 + 1), max(0, c - 1), min(demarr_cl.shape[1], c + 1 + 1)
        step_size = search_win_size // 2
        # define searching window boundaries
        up, dwn, lf, rit = max(0, r - step_size), min(demarr_cl.shape[0], r + step_size + 1), \
                           max(0, c - step_size), min(demarr_cl.shape[1], c + step_size + 1)
        curr_block = demarr_cl[up:dwn, lf:rit].copy()   #search window, of 9x9 size by default, with 555 in current pt
        demarr_cl[r, c] = np.inf    # mark searched point as inf
        while self.non_stopping_criteria(demarr_cl[up0:dwn0, lf0:rit0]):
            ri, ci = self.search_next_pt(curr_block, search_win_size)    # current point = 555
            # update current row, column no.
            r, c = r + ri, c + ci
            line_ls.append(rc2coords(self.dem_transform, (r, c)))
            up0, dwn0, lf0, rit0 = max(0, r - 1), min(demarr_cl.shape[0], r + 1 + 1), \
                               max(0, c - 1), min(demarr_cl.shape[1], c + 1 + 1)    # update block boundaries, 3x3
            up, dwn, lf, rit = max(0, r - step_size), min(demarr_cl.shape[0], r + step_size + 1), \
                               max(0, c - step_size), min(demarr_cl.shape[1], c + step_size + 1)    # update searching window boundaries
            demarr_cl[r, c] = 555
            curr_block = demarr_cl[up:dwn, lf:rit].copy()   # with 555 in current pt
            demarr_cl[r, c] = np.inf    # mark searched point as inf

        if np.any(np.isnan(demarr_cl[up0:dwn0, lf0:rit0])):   # add the boundary cell at last if exists
            id_bd = np.argwhere(np.isnan(demarr_cl[up0:dwn0, lf0:rit0]))[0]
            line_ls.append(rc2coords(self.dem_transform, (id_bd[0]+up0, id_bd[1]+lf0)))
            self.countline[0] += 1
        else:
            for stop_val in self.stopping_vals:
                if np.any(demarr_cl[up0:dwn0, lf0:rit0] == stop_val):   # add the iwl cell at last if exists
                    id_iwl = np.argwhere(demarr_cl[up0:dwn0, lf0:rit0] == stop_val)[0]
                    line_ls.append(rc2coords(self.dem_transform, (id_iwl[0]+up0, id_iwl[1]+lf0)))
                    self.countline[0] += 1
                    break
        return line_ls

    def non_stopping_criteria(self, blockarr):
        criterion_1 = bool(~np.isinf(np.nanmin(blockarr)))   # the min is not inf; there is at least one inf in block
        criterion_2 = True  # not stopping
        for stop_val in self.stopping_vals:
            criterion_2 = criterion_2 & (~np.any(blockarr == stop_val))    # not reaching the grids with stop signs
        criterion_3 = ~np.any(np.isnan(blockarr))    # the block is hitting the boundary of model domain; Sea-side includes
        return bool(criterion_1 & criterion_2 & criterion_3)

    def search_next_pt(self, blockarr, search_win_size):
        # calc 3x3 min
        resample_shape = (3, 3)
        sh = resample_shape[0], search_win_size // 3, resample_shape[1], search_win_size // 3
        blockarr_resample = np.nanmin(np.nanmin(blockarr.reshape(sh), axis=-1), axis=1)

        # locate min direction, determine candidate for next step
        direc_idx = np.argwhere(blockarr_resample == np.nanmin(blockarr_resample))[-1]
        candidates = np.array([[direc_idx[0]-1, direc_idx[1]],
                               [direc_idx[0]+1, direc_idx[1]],
                               [direc_idx[0], direc_idx[1]],
                               [direc_idx[0], direc_idx[1]-1],
                               [direc_idx[0], direc_idx[1]+1]])
        candidates = candidates[~((candidates < 0) | (candidates > 2)).any(axis=1)]
        blockarr_central = blockarr[search_win_size//2-1:search_win_size//2+2,
                           search_win_size//2-1:search_win_size//2+2].copy()    # central block always 3x3
        candidates_val = [blockarr_central[r, c] for r, c in candidates]
        ids_min = candidates[np.argwhere(candidates_val == np.nanmin(candidates_val))[-1][0]]   # -1: taking backward first min
        ids_curr_pt = np.argwhere(blockarr_central == 555)[0]  # array([r,c]) of the current point
        ri = ids_min[0] - ids_curr_pt[0]   # row no. increase; -1, 0, 1
        ci = ids_min[1] - ids_curr_pt[1]   # column no. increase; -1, 0, 1
        if ((ri == 0) & (ci == 0)) | np.all(np.isinf(candidates_val)):
            ids_min = np.argwhere(blockarr_central == np.nanmin(blockarr_central))[-1]
            ri = ids_min[0] - ids_curr_pt[0]   # row no. increase; -1, 0, 1
            ci = ids_min[1] - ids_curr_pt[1]   # column no. increase; -1, 0, 1
        return ri, ci

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

    # main post-processing function
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

    def update_drainage_map(self, trajs, success_pts):
        """ Drainage_map: contain integers. 0 for not drainage, 1, + for the number+1 of successful trajs """
        self.drainage_map = np.zeros(self.demarr.shape)
        for success_pt_idx in range(len(success_pts)):
            rcs = np.array(
                [list(coords2rc(self.dem_transform, coords)) for coords in trajs[success_pts[success_pt_idx]]]).astype(
                int)
            self.drainage_map[rcs[:, 0], rcs[:, 1]] = success_pt_idx + 1
        return 0

    def sep_trajs(self, trajs):
        success_pts = []
        failed_pts = []
        for key in trajs.keys():
            r, c = coords2rc(self.dem_transform, trajs[key][-1])
            if self.is_not_stopping_grids(self.demarr[r, c]):
                failed_pts.append(key)
            else:  # hit IWL value or boundary
                success_pts.append(key)
        return success_pts, failed_pts

    def eliminate_clusters_in_trajs(self, trajs, starting_pts):
        """ backward search, connect with the most downstream point of over-3-step-upstream points within 3x3 area if exists. """
        curving_threshold = self.straighten_degree
        for stpt in starting_pts:
            curr_traj = trajs[stpt].copy()
            rcs_ls = [list(coords2rc(self.dem_transform, coords)) for coords in curr_traj]
            pt_id = len(curr_traj) - 1
            while pt_id > curving_threshold:
                r, c = rcs_ls[pt_id]
                rcs = np.array(rcs_ls[:pt_id - curving_threshold]).astype(
                    int)  # restrict to upstream points; next, filter for 3x3 area
                block_check = rcs[
                              (rcs[:, 0] >= r - 1) & (rcs[:, 0] <= r + 1) & (rcs[:, 1] >= c - 1) & (rcs[:, 1] <= c + 1),
                              :]
                if block_check.size > 0:
                    next_pt_id = rcs_ls.index(list(block_check[0, :]))  # [0] most upstream, [-1] most downstream
                    del curr_traj[next_pt_id + 1:pt_id]
                    pt_id = next_pt_id
                else:
                    pt_id -= 1
            trajs[stpt] = curr_traj
        return trajs

    def connect_failed_success_if_possible(self, trajs):
        """ Connect failed trajectories to succeeded trajectories if they share points. """
        success_pts, failed_pts = self.sep_trajs(trajs)
        self.update_drainage_map(trajs, success_pts)
        for failed_pt in failed_pts:
            rcs = np.array([list(coords2rc(self.dem_transform, coords)) for coords in trajs[failed_pt]]).astype(int)
            drainage_tags = self.drainage_map[rcs[:, 0], rcs[:, 1]].astype(int)
            if np.any(drainage_tags):  # traj sharing any point with successful trajs
                shared_idx = np.nonzero(drainage_tags)[0][0]
                # 1D array, first item, the index of the first shared pt (row) in the rcs list
                shared_coords = rc2coords(self.dem_transform, rcs[shared_idx, :])
                new_traj = trajs[failed_pt][:trajs[failed_pt].index(shared_coords)]
                new_traj.extend(trajs[success_pts[drainage_tags[shared_idx] - 1]][
                                trajs[success_pts[drainage_tags[shared_idx] - 1]].index(shared_coords):])
                trajs[failed_pt] = new_traj
                trajs = self.eliminate_clusters_in_trajs(trajs, [failed_pt])
        return trajs

    def find_boundary_min(self, rcs):
        boundary_min_dems = np.empty(rcs.shape[0])
        boundary_min_dems.fill(np.nan)
        for i in range(rcs.shape[0]):
            r, c = rcs[i, :]
            block_check = rcs[(rcs[:, 0] >= r - 1) & (rcs[:, 0] <= r + 1) & (rcs[:, 1] >= c - 1) & (rcs[:, 1] <= c + 1),
                          :].astype(int)
            if block_check.size / 2 < 9:  # at boundary
                dem_arr = self.demarr[r - 1:r + 2, c - 1:c + 2].copy()
                dem_arr[block_check[:, 0] - (r - 1), block_check[:, 1] - (c - 1)] = np.nan
                boundary_min_dems[i] = np.nanmin(dem_arr)
        min_rc_idx = np.where(boundary_min_dems == np.nanmin(boundary_min_dems))[0][0]
        return rcs[min_rc_idx, :]

    def continue_failed_trajs(self, trajs):
        success_pts, failed_pts = self.sep_trajs(trajs)
        self.update_drainage_map(trajs, success_pts)
        for failed_pt in failed_pts:  # only self-grounded trajs exist
            # saddle-point-driven reviving trajs
            rcs = np.array([list(coords2rc(self.dem_transform, coords)) for coords in trajs[failed_pt]])
            r, c = coords2rc(self.dem_transform, trajs[failed_pt][-1])
            masked_dem_arr = self.demarr.copy()
            masked_dem_arr[rcs[:, 0].astype(int), rcs[:, 1].astype(int)] = np.inf  # mark as 'searched'
            # Hard loop until successfully constructed all the failed lines; loop_count for maximum looping time control
            loop_count = 0
            while self.is_not_stopping_grids(self.demarr[r, c]):
                loop_count += 1
                # if loop_count > 10:  # Manual termination setting: Maximum 10 times of looping
                #     break
                r, c = self.find_boundary_min(rcs)
                curr_coords = rc2coords(self.dem_transform, (r, c))
                extended_line = self.single_line_search(curr_coords,
                                                        self.default_search_win_size, modified_dem=masked_dem_arr)
                extended_rcs = np.array(
                    [list(coords2rc(self.dem_transform, coords)) for coords in extended_line]).astype(int)
                masked_dem_arr[extended_rcs[:, 0], extended_rcs[:, 1]] = np.inf  # mark as 'searched'
                # trajs[failed_pt] = trajs[failed_pt][:trajs[failed_pt].index(curr_coords) + 1] + extended_line
                trajs[failed_pt] = trajs[failed_pt] + extended_line
                rcs = np.array([list(coords2rc(self.dem_transform, coords)) for coords in trajs[failed_pt]])
                r, c = coords2rc(self.dem_transform, trajs[failed_pt][-1])
            else:  # successful, check with existing successful trajs & post-process clusters
                # try connect to previously succeeded trajs
                drainage_tags = self.drainage_map[rcs[:, 0], rcs[:, 1]].astype(int)
                if np.any(drainage_tags):  # traj sharing any point with successful trajs
                    shared_idx = np.nonzero(drainage_tags)[0][0]
                    # 1D array, first item, the index of the first shared pt (row) in the rcs list
                    shared_coords = rc2coords(self.dem_transform, rcs[shared_idx, :])
                    new_traj = trajs[failed_pt][:trajs[failed_pt].index(shared_coords)]
                    new_traj.extend(trajs[success_pts[drainage_tags[shared_idx] - 1]][
                                    trajs[success_pts[drainage_tags[shared_idx] - 1]].index(shared_coords):])
                    trajs[failed_pt] = new_traj
                trajs = self.eliminate_clusters_in_trajs(trajs, [failed_pt])
        return trajs

    def continue_failed_river_trajs(self, trajs):
        success_pts, failed_pts = self.sep_trajs(trajs)
        for failed_pt in failed_pts:  # only self-grounded trajs exist
            # saddle-point-driven reviving trajs
            rcs = np.array([list(coords2rc(self.dem_transform, coords)) for coords in trajs[failed_pt]])
            r, c = coords2rc(self.dem_transform, trajs[failed_pt][-1])
            masked_dem_arr = self.demarr.copy()
            masked_dem_arr[rcs[:, 0].astype(int), rcs[:, 1].astype(int)] = np.inf  # mark as 'searched'
            # Hard loop until successfully constructed all the failed lines; loop_count for maximum looping time control
            loop_count = 0
            while self.is_not_stopping_grids(self.demarr[r, c]):
                loop_count += 1
                # if loop_count > 10:  # Manual termination setting: Maximum 10 times of looping
                #     break
                r, c = self.find_boundary_min(rcs)
                curr_coords = rc2coords(self.dem_transform, (r, c))
                extended_line = self.single_line_search(curr_coords,
                                                        self.default_search_win_size, modified_dem=masked_dem_arr)
                extended_rcs = np.array(
                    [list(coords2rc(self.dem_transform, coords)) for coords in extended_line]).astype(int)
                masked_dem_arr[extended_rcs[:, 0], extended_rcs[:, 1]] = np.inf  # mark as 'searched'
                # trajs[failed_pt] = trajs[failed_pt][:trajs[failed_pt].index(curr_coords) + 1] + extended_line
                trajs[failed_pt] = trajs[failed_pt] + extended_line
                rcs = np.array([list(coords2rc(self.dem_transform, coords)) for coords in trajs[failed_pt]])
                r, c = coords2rc(self.dem_transform, trajs[failed_pt][-1])
            else:
                trajs = self.eliminate_clusters_in_trajs(trajs, [failed_pt])
        return trajs

    def post_process_initial_trajs(self, read_from_shp=False, shpfile=None, print_counting_report=True):
        if read_from_shp:
            if shpfile is None:
                shpfile = f"{self.work_dir}trajs_initial_search_from_starting_points.shp"
            trajs = self.read_shp_to_trajs(shpfile)
        else:
            trajs = self.initial_stpts_trajs(save_to_shp=True)
        success_trajs, _ = self.sep_trajs(trajs)
        trajs = self.eliminate_clusters_in_trajs(trajs, success_trajs)
        trajs = self.connect_failed_success_if_possible(
            trajs)  # newly succeeded trajs are cluster-free (eliminated inside)
        trajs = self.continue_failed_trajs(trajs)
        trajs = self.connect_failed_success_if_possible(
            trajs)  # newly succeeded trajs are cluster-free (eliminated inside)

        if print_counting_report:
            success_trajs, failed_trajs = self.sep_trajs(trajs)
            self.countline.append(len(success_trajs))
            self.countline.append(len(failed_trajs))
            print('The increased number of successful trajectories after post-process: ', self.countline[1])
            print('The total number of failed trajectories after post-process: ', self.countline[2])
        return trajs

    def define_main_river(self, print_counting_report=True):
        trajs = self.initial_stpts_trajs(save_to_shp=False)
        trajs = self.continue_failed_river_trajs(trajs)

        if print_counting_report:
            success_trajs, failed_trajs = self.sep_trajs(trajs)
            self.countline.append(len(success_trajs))
            print('The number of successful trajectories: ', self.countline[1])
        return trajs

    # external calls
    def generate_drainage_shp(self, outshpfile, initial_trajs_saved=False):
        trajs = self.post_process_initial_trajs(read_from_shp=initial_trajs_saved)
        self.save_to_shp_lines(trajs, outshpfile)
        return print('Successfully saved drainage network shapefile.')

    def generate_drainage_trajs(self, initial_trajs_saved=False):
        trajs = self.post_process_initial_trajs(read_from_shp=initial_trajs_saved)
        return trajs

    def generate_main_river_shp(self, outshpfile, dem_aggregate_factor=3, dem_aggregate_func=np.nanmean):
        self.demarr, self.dem_transform = aggregate_arr(self.demarr, self.dem_transform,
                                                        dem_aggregate_factor, agg_func=dem_aggregate_func)
        self.demarr[np.isnan(self.demarr)] = 999
        trajs = self.define_main_river()
        self.save_to_shp_lines(trajs, outshpfile)
        return print('Successfully saved main river shapefile.')


def perform_SDR(work_dir, starting_pts, dem_tif_file, out_line_shp_file, stopping_categories,
                sdr_search_window_size=9, continueRun=False):
    directory_checking(work_dir)
    Traj_search(work_dir, starting_pts, dem_tif_file, stoping_categories=stopping_categories,
                search_win_size=sdr_search_window_size).generate_drainage_shp(out_line_shp_file,
                                                                              initial_trajs_saved=continueRun)
    return 0

