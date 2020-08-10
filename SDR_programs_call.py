from SDR_algorithm import *
from SDR_algorithm_filter import *
from SDR_algorithm_reco import reconstruct_flood_inundation_map


def sdr_Searching(maximum_extent_inundation_map, time_step_of_maximum_if_nc_file_used, 
				  dem_tif_file, stopping_values_in_dem, 
				  results_dir, output_sdr_shapefile_name, 
				  customised_searching_win_size=9, initial_run_bool=True, 
				  require_representative_thalweg_selection_bool=True, rep_traj_ratio=1/200):
	"""
	This function is used for SDR Thalwegs configuration ONLY. For mainstream centroid line definition, use sdr_Searching_MCL.
	Inputs:
			maximum_extent_inundation_map: Geotiff format, non-flooded area marked as nan
			time_step_of_maximum_if_nc_file_used: set to 0 if geotiff file is provided. 
						For NetCDF file, specify the layer number from the first dimension.
			dem_tif_file: DEM file in geotiff format. This file will also determine searching domain. 
						Mask unwanted area with np.nan. Specify termination zones with termination values.
			stopping_values_in_dem: termination values set in DEM file. A list of values, e.g. [-555, -666], or [-666] for a single value.
			results_dir: directory for saving results, e.g. 'Directory_to_results/'
			output_sdr_shapefile_name: e.g. 'SDR_searching_results.shp'
			customised_searching_win_size: searching window size in SDR-Searching, default=9
			initial_run_bool: default=True, Specify if this is your first time run with True, not first run with False
						After the first time run, a file named 'trajs_initial_search_from_starting_points.shp' is generated. By using 
						this file in the successive runs can save some running time during parameter tuning.
			require_representative_thalweg_selection_bool: default=True, to perform representative thalwegs selection
			rep_traj_ratio: default=1/200, the ratio used to select representative thalweg from each thalweg group
	Outputs:
			SDR Thalwegs in a Line Shapefile(ESRI)
	"""
	directory_checking(results_dir)
	stpts = get_starting_pts_from_maximum_inundation_extent(maximum_extent_inundation_map, 
															time_step_of_maximum_if_nc_file_used)
	dem_tif_file = dem_checking(dem_tif_file)
	if require_representative_thalweg_selection_bool:
		starttime = time()
		perform_SDR(results_dir, stpts, dem_tif_file, 
					f'{results_dir}{output_sdr_shapefile_name[:-4]}_before_rep_thal_selec.shp', 
					stopping_values_in_dem, customised_searching_win_size, (not initial_run_bool))
		Select_Rep_pts(results_dir, dem_tif_file, 
					   f'{results_dir}{output_sdr_shapefile_name[:-4]}_before_rep_thal_selec.shp'
					   ).save_rep_trajs_2shp_with_this_ratio(output_sdr_shapefile_name, rep_traj_ratio)
		print(f"Used time for SDR-Searching: {time() - starttime}s")
	else:
		starttime = time()
		perform_SDR(results_dir, stpts, dem_tif_file, output_sdr_shapefile_name, 
					stopping_values_in_dem, customised_searching_win_size, (not initial_run_bool))
		print(f"Used time for SDR-Searching: {time() - starttime}s")
	return 0


def sdr_Searching_MCL(starting_coordinates, dem_tif_file, stopping_values_in_dem,
					  results_dir, output_sdr_shapefile_name,  
					  dem_aggregate_rate=3, dem_aggregate_function=np.nanmean, 
					  customised_searching_win_size=9):
	"""
	This function is used for mainstream centroid line definition ONLY. For SDR_Thalwegs configuration, use sdr_Searching.
	Inputs: 
			starting_coordinates: coordinates (x_coord, y_coord)
			dem_tif_file: DEM file in geotiff format. This file will also determine searching domain. 
						Mask unwanted area with np.nan. Specify termination zones with termination values.
			stopping_values_in_dem: termination values set in DEM file. A list of values, e.g. [-555, -666], or [-666] for a single value.
			results_dir: directory for saving results, e.g. 'Directory_to_results/'
			output_sdr_shapefile_name: e.g. 'SDR_searching_results.shp'
			dem_aggregate_rate: default=3, aggregate DEM data with the factor of 3, e.g. 20m resolution to 60m
			dem_aggregate_function: default=np.nanmean, the function used to resample DEM in aggregation, e.g. np.nanmin, np.nanmax, etc.
			customised_searching_win_size: searching window size in SDR-Searching, default=9
	Outputs:
			Mainstream centroid line in a Line Shapefile(ESRI)
	"""
	directory_checking(results_dir)
	dem_tif_file = dem_checking(dem_tif_file)
	stpt = np.array(starting_coordinates).reshape(1, 2)
	starttime = time()
    Traj_search(results_dir, stpt, dem_tif_file, stopping_values_in_dem, customised_searching_win_size
				).generate_main_river_shp(output_sdr_shapefile_name, dem_aggregate_rate, dem_aggregate_function)
    print(f"Used time for MCL definition: {time() - starttime}s")
	return 0


def sdr_RL(mcl_file, sdr_thalwegs_file, resample_rate_mcl, resample_rate_sdr_thalwegs, 
		   results_dir, output_rls_file_name):
	"""
	This function is to extract representative locations for DL modelling from sdr_Searching and sdr_Searching_MCL results.
	Inputs: 
			input_dir: directory for saved input SDR thalwegs and MCL files, e.g. 'Directory_to_inputs/'
			mcl_file: the mainstream centroid line shapefile generated using sdr_Searching_MCL
			sdr_thalwegs_file: the SDR Thalwegs shapefile generated using sdr_Searching
			resample_rate_mcl: the resampling distance along the MCL line
			resample_rate_sdr_thalwegs: the proportion of RLs to be selected from Thalweg-RLs
			results_dir: directory for saving results, e.g. 'Directory_to_results/'
			output_rls_file_name: e.g. 'SDR_rls_results.shp'
	Outputs:
			RLs in a Point Shapefile(ESRI)
	"""
	starttime = time()
	extract_rep_pts(f'{input_dir}{mcl_file}', f'{input_dir}{sdr_thalwegs_file}', 
					resample_rate_mcl, resample_rate_sdr_thalwegs, 
					results_dir, output_rls_file_name)
	print(f"Used time for SDR-RL: {time() - starttime}s")
	return 0


def sdr_Reco(dem_tif_file, targeted_x_coords, targeted_y_coords, 
			 rls_shp_file, sdr_thalwegs_file, water_levels_at_rls, 
			 tidal_boundary_grids_tif_file=None, sealvl=None, 
			 save_to_tif_file=None, results_dir=None, reference_tif_file=None):
	"""
	This function is to reconstruct the flood inundation map from water level information at RLs.
	Inputs: 
			dem_tif_file: DEM file in geotiff format. This file will also determine reconstruction domain. 
			targeted_x_coords, targeted_y_coords: numpy arrays contain the x/y coordinates for a gridded network space
						e.g. np.array([1,2,3]), np.array([1,2,3]) will produce a gridded space of shape 3x3
			rls_shp_file: the SDR-RL shapefile generated using sdr_RL
			sdr_thalwegs_file: the SDR Thalwegs shapefile generated using sdr_Searching
			water_levels_at_rls: an array of water levels corresponding to each RL in the order of FID in rl_shp_file
			tidal_boundary_grids_tif_file: default=None, 
						if provided, the grids having value=1 in this tif file will be recognized as downstream boundary
			sealvl: if tidal_boundary_grids_tif_file is provided, the water level at the DB needs to be provided to sealvl
			save_to_tif_file: default=None, if needs to save the reconstructed flood inundation map to a file, specify file name here
			results_dir: directory for saving results, e.g. 'Directory_to_results/'
			reference_tif_file: the reference geotiff file from where the map extent, transform and projection will be used.
						it can be the dem_tif_file.
	Outputs:
			One from two options:
				1. A 2D numpy array with reconstructed water levels at targeted locations;
				2. A GeoTiff file of the reconstructed flood inundation map.
	"""
	if save_to_tif_file is None:
		starttime = time()
		output_arr = reconstruct_flood_inundation_map(dem_tif_file, targeted_x_coords, targeted_y_coords,
													  rls_shp_file, sdr_thalwegs_file, water_levels_at_rls, 
													  tidal_boundary_grids_tif_file, sealvl)
		print(f"Used time for SDR-Reco: {time() - starttime}s")
		return output_arr
	else:
		assert (results_dir is not None), 'Plesae provide a directory for saving results, e.g. results_dir='
		directory_checking(results_dir)
		assert (reference_tif_file is not None), 'Please provide a referece GeoTiff file at reference_tif_file=, to set the basic information of the output tif file.'
		dem_tif_file = dem_checking(dem_tif_file)
		starttime = time()
		reconstruct_flood_inundation_map(dem_tif_file, targeted_x_coords, targeted_y_coords,
										 rls_shp_file, sdr_thalwegs_file, water_levels_at_rls, 
										 tidal_boundary_grids_tif_file, sealvl, 
										 f"{results_dir}{save_to_tif_file}", reference_tif_file)
		print(f"Used time for SDR-Reco: {time() - starttime}s")
		return 0

