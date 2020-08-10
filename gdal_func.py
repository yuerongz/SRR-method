from osgeo import gdal, ogr, gdalconst
import numpy as np
import geopandas


def coords2rc(transform, coords):
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]
    col = int((coords[0] - xOrigin) / pixelWidth)
    row = int((yOrigin - coords[1]) / pixelHeight)
    return row, col


def rc2coords(transform, rc):
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = transform[5]
    coordX = xOrigin + pixelWidth * (rc[1] + 0.5)
    coordY = yOrigin + pixelHeight * (rc[0] + 0.5)
    return coordX, coordY


def read_shp_point(filename):
    """Read the Point shapefile attributes as a list of xy coordinates"""
    result = list()
    startPoints = geopandas.read_file(filename)
    for pt in startPoints.geometry:
        result.append((pt.x, pt.y))
    return result


def read_shp_line(filename):
    """read multi-line shapefile as point coords which defining the lines"""
    output = list()
    lineshp = geopandas.read_file(filename)
    for ln in lineshp.geometry:
        output.append(list(zip(ln.xy[0], ln.xy[1])))
    return output


def gdal_asarray(rasterfile):
    ds = gdal.Open(rasterfile)
    band = ds.GetRasterBand(1)
    return band.ReadAsArray()


def gdal_transform(rasterfile):
    return gdal.Open(rasterfile).GetGeoTransform()


def gdal_writetiff(arr_data, outfile, ras_temp):
    ds = gdal.Open(ras_temp)
    driver = gdal.GetDriverByName('GTiff')
    outdata = driver.Create(outfile, arr_data.shape[1], arr_data.shape[0], 1, gdal.GDT_Float32)
    outdata.SetGeoTransform(ds.GetGeoTransform())  ##sets same geotransform as temp
    outdata.SetProjection(ds.GetProjection())  ##sets same projection as temp
    outband = outdata.GetRasterBand(1)
    outband.WriteArray(arr_data)
    outdata.FlushCache()  ##saves to disk!!
    return print('gdal_writeriff successful.')


def raster_aggregate(srcfile, outfile, aggregate_level):
    src = gdal.Open(srcfile, gdalconst.GA_ReadOnly)
    dst_gt = list(src.GetGeoTransform())
    dst_gt[1] = dst_gt[1] * aggregate_level
    dst_gt[5] = dst_gt[5] * aggregate_level
    dst_gt = tuple(dst_gt)
    width = int(src.RasterXSize / aggregate_level)
    height = int(src.RasterYSize / aggregate_level)
    dst = gdal.GetDriverByName('GTiff').Create(outfile, width, height, 1, gdalconst.GDT_Float32)
    dst.SetGeoTransform(dst_gt)
    dst.SetProjection(src.GetProjection())
    gdal.ReprojectImage(src, dst, src.GetProjection(), None, gdalconst.GRA_Min)
    return 0



