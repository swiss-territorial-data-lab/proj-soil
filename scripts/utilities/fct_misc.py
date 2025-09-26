import os
import sys
from rasterio.features import shapes
from rasterio.features import rasterize
from shapely.geometry import shape
import geopandas as gpd


def format_logger(logger, LOG_FILE):
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(LOG_FILE, level="INFO")

    return logger

def raster2gdf(raster_data, transform, value_field, mask=None):
    """
    Convert a raster file to a GeoDataFrame.
    
    Parameters:
    raster_data (numpy.ndarray): Raster data
    transform (Affine): Affine transform of the raster.
    value_field (str): Name of the field to store raster values.
    mask (numpy.ndarray, optional): Mask array to filter specific values.
    
    Returns:
    GeoDataFrame: A GeoDataFrame with polygon geometries and raster values.
    """
    # with rasterio.open(raster_path) as src:
    #     # Read the raster data
    #     raster_data = src.read(1)  # Read the first band (modify for multi-band rasters)
    #     transform = src.transform  # Get the transform information
        
        # Vectorize the raster using rasterio.features.shapes
    shapes_gen = shapes(raster_data, transform=transform, mask=mask)
    
    # Convert shapes to GeoDataFrame
    records = []
    for geom, value in shapes_gen:
        records.append({
            'geometry': shape(geom),
            value_field: value
        })
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(records, crs='2056')
    
    return gdf

def gdf2raster(gdf, value_field, out_shape, transform, nodata=0, dtype='int32'):
    """
    Rasterize a GeoDataFrame to a raster array.
    
    Parameters:
    gdf (GeoDataFrame): The GeoDataFrame containing geometries to rasterize.
    value_field (str): The column in the GeoDataFrame to use as raster values.
    out_shape (tuple): Shape of the output raster (rows, cols).
    transform (Affine): Affine transform for the raster.
    nodata (numeric): Value to assign to cells not covered by geometries.
    dtype (str): Data type of the output raster array.
    
    Returns:
    numpy.ndarray: A rasterized array.
    """
    # Prepare the shapes (geometry, value) for rasterizing
    shapes = ((geom, row[value_field]) for geom, row in zip(gdf.geometry, gdf.itertuples()))
    
    # Rasterize
    raster = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=nodata,
        dtype=dtype
    )
    
    return raster
