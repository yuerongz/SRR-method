from gdal_func import rc2coords, gdal_asarray
from nc_func import *


def get_boundary_pts(arr, nan_threshold):
    arr_01 = arr.copy()
    arr_01[~np.isnan(arr_01)] = 1
    arr_01[np.isnan(arr_01)] = 0    # nan=0
    checking_arrs = [arr_01[0:-2, 0:-2], arr_01[0:-2, 1:-1], arr_01[0:-2, 2:],
                     arr_01[1:-1, 0:-2], arr_01[1:-1, 1:-1], arr_01[1:-1, 2:],
                     arr_01[2:,   0:-2], arr_01[2:,   1:-1], arr_01[2:,   2:]]
    count_arr = checking_arrs[0] + checking_arrs[1] + checking_arrs[2] + checking_arrs[3] + checking_arrs[5] + \
                checking_arrs[6] + checking_arrs[7] + checking_arrs[8]
    count_arr[count_arr >= nan_threshold] = 0    # valid cells: count between 1~3
    target_arr = count_arr * checking_arrs[4]
    target_arr[target_arr != 0] = 1     # located cells for starting with value 1 (others 0)
    target_arr = np.r_[[np.zeros(target_arr.shape[1])], target_arr, [np.zeros(target_arr.shape[1])]]   # add row at beginning & end
    target_arr = np.c_[np.zeros(target_arr.shape[0]), target_arr, np.zeros(target_arr.shape[0])]   # add column at beginning & end
    return target_arr


def get_pts_coords_from_agg(eventfile, maxtime, nanthreshold=4):
    """ get the central points of all the 1 cells in the aggregated raster"""
    if eventfile[-2:] == 'nc':
        ncdata = Dataset(eventfile)
        maxarr = ncarr2gdalarr(ncdata.variables['water_level'][maxtime*4, :, :].data)
        maxarr[maxarr == -999] = np.nan
        arr_agg, agg_trans = maxarr, getNCtransform(ncdata)
    elif eventfile[-3:] == 'tif':
        arr_agg = gdal_asarray(eventfile)
        agg_trans = gdal.Open(eventfile).GetGeoTransform()
    else:
        raise TypeError('The file type of the file for starting points selection cannot be recognized. Only NetCDF/.nc or GeoTiff/.tif can be used.')
    boundary_pts = get_boundary_pts(arr_agg, nanthreshold)
    idx = np.array(list(np.where(boundary_pts == 1)))
    xy_coords = np.zeros(idx.shape)
    for i in range(idx.shape[1]):
        xy_coords[:, i] = rc2coords(agg_trans, idx[:, i])
    return xy_coords


def get_starting_pts_from_maximum_inundation_extent(eventfile, maxtime, nanthreshold=4):
    """
    Inputs:
        eventfile: the file directory/filename of the maximum inundation map.
                   Only NetCDF/.nc or GeoTiff/.tif format can be recognized
                   The non-inundated area needs to be marked as nan; other area being any number other than nan.
        maxitime: if .nc file is provided, indicate the layer in the first dimension to use, count starts from 0;
                   If .tif format is used, input 0.
        nanthreshold: default=4, the threshold used to identify boundary grids as described in our paper
    Output:
        a numpy array of shape (N, 2), N number of coordinates (x, y)
    """
    return np.array(get_pts_coords_from_agg(eventfile, maxtime, nanthreshold)).T

