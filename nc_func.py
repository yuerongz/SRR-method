from osgeo import gdal
import numpy as np
from netCDF4 import Dataset


def getNCtransform(dataset):
    xres = dataset.variables['x'][1].data - dataset.variables['x'][0].data
    xmin = dataset.variables['x'][0].data - xres / 2  # calculate the original extent xmin
    yres = dataset.variables['y'][-1].data - dataset.variables['y'][-2].data
    ymax = dataset.variables['y'][-1].data + yres / 2  # calculate the original extent ymax
    geotransform = (xmin, xres, 0, ymax, 0, -yres)
    return geotransform


def ncarr2gdalarr(arr, no_data=-999):
    gdalarr = arr.copy()
    gdalarr[gdalarr==no_data] = np.nan
    return np.flip(gdalarr, 0)


if __name__ == '__main__':
    print('import desired function!')