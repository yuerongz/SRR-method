# Created by Yuerong Zhou on 28th July 2020
# This is a template of applying the SDR method introduced in out MethodsX paper 
# The SDR method consists of mainly three functional programs: SDR-Searching, SDR-RL and SDR-Reco
# Examples on how to apply these programs to your dataset are provided below: 

#### Step 1: configuring SDR Thalwegs and mainstream centroid line 
from SDR_programs_call import sdr_Searching, sdr_Searching_MCL
sdr_Searching(maximum_extent_inundation_map, time_step_of_maximum_if_nc_file_used, 
			  dem_tif_file, stopping_values_in_dem, 
			  results_dir, output_sdr_thalwegs_file_name, 
			  customised_searching_win_size=9, initial_run_bool=True, 	# optional parameters
			  require_representative_thalweg_selection_bool=True, rep_traj_ratio=1/200)		# optional parameters
# to obtain the SDR Thalwegs shapefile: sdr_thalwegs_file_name

sdr_Searching_MCL(starting_coordinates, dem_tif_file, stopping_values_in_dem,
				  results_dir, output_mcl_file_name,  
				  dem_aggregate_rate=3, dem_aggregate_function=np.nanmean, 		# optional parameters
				  customised_searching_win_size=9)		# optional parameters
# to obtain the mainstream centroid line file: mcl_file_name


#### Step 2: Selecting representative locations (RLs) in the model domain 			  
from SDR_programs_call import sdr_RL
sdr_RL(input_dir, mcl_file_name, sdr_thalwegs_file_name, 
	   resample_rate_mcl, resample_rate_sdr_thalwegs, 
	   results_dir, output_rls_file_name)
# to obtain the RLs point shapefile: rls_file_name


#### Step 3: Reconstructing the flood inundation map in the model domain based on modelled water levels at RLs
####		 Note: the SDR method itself does not includes the water level modelling at RLs.
####			   Users need to use an external model to simulate the water levels from 
####               hydraulic system boundary conditions, such as deep learning models (Reference to our work).
from SDR_programs_call import sdr_Reco
sdr_Reco(dem_tif_file, targeted_x_coords, targeted_y_coords, 
		 rls_shp_file, sdr_thalwegs_file, water_levels_at_rls, 
		 tidal_boundary_grids_tif_file=None, sealvl=None,  		# optional parameters
		 save_to_tif_file=None, results_dir=None, reference_tif_file=None) 		# optional parameters
# to obtain reconstructed flood inundation map from water levels of RLs

