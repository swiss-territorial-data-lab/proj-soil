
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module='shapely')

import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box

from loguru import logger
import yaml


def clip_tiffs(SOURCE_FOLDER, TARGET_FOLDER, CLIP_SHAPEFILE):
    """
    Clip tiff files in the specified folder using the provided shapefile.
    
    Parameters:
        SOURCE_FOLDER (str): Path to the source folder containing raster files.
        TARGET_FOLDER (str): Path to the target folder to save the clipped raster files.
        CLIP_SHAPEFILE (str): Path to the shapefile used for clipping.
    
    Returns:
        None
    """
    # Read the shapefile for clipping
    mask_gdf = gpd.read_file(CLIP_SHAPEFILE, crs="EPSG:2056")

    # Iterate over the files in the source folder
    for file in os.listdir(SOURCE_FOLDER):
        if not file.endswith(("tif", "tiff")):
            continue

        logger.info("--------")
        logger.info(f"Clipping {file}")
        
        # Open the source raster file
        source_file_path = os.path.join(SOURCE_FOLDER, file)
        with rasterio.open(source_file_path) as src:

            # Read the file profile
            srcprof = src.profile.copy()
            
            # Create a GeoDataFrame for the source raster bounds
            src_gdf = gpd.GeoDataFrame(geometry=[box(*src.bounds)], crs="EPSG:2056")

            # Clip the source GeoDataFrame with the mask GeoDataFrame
            clipped_gdf = gpd.clip(src_gdf, mask_gdf)

            if clipped_gdf.empty:
                logger.info(f"Empty intersection for {file}")
                continue

            # Calculate the window of the intersection extent
            masked_ar, masked_transf = mask(src, clipped_gdf.geometry, crop=True)
            
            # Update the profile with the new window and transform
            srcprof.update({
                'height': masked_ar.shape[1],
                'width': masked_ar.shape[2],
                'transform': masked_transf
            })

            # Create and save the clipped raster file
            with rasterio.open(os.path.join(TARGET_FOLDER, file), "w", **srcprof) as tgt:
                tgt.write(masked_ar)

if __name__ == "__main__":
    # Argument and parameter specification
    parser = argparse.ArgumentParser(
        description="The script clips tiff files using a shapefile")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        default="/Users/nicibe/Desktop/Job/swisstopo_stdl/soil_fribourg/proj-soils/config_for_docker/config-infere_OFS.yaml"
        )
    args = parser.parse_args()

    # load input parameters
    with open(args.config_file) as fp:
        cfg = yaml.load(
            fp,
            Loader=yaml.FullLoader)[os.path.basename(__file__)]

    SOURCE_FOLDER = cfg["source_folder"]
    TARGET_FOLDER = cfg["target_folder"]
    CLIP_SHAPEFILE = cfg["clip_shapefile"]
    LOG_FILE = cfg["log_file"]

    # set up logger
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")
    logger.add(LOG_FILE, level="INFO")

    logger.info("---------------------")
    logger.info(f"{SOURCE_FOLDER = }")
    logger.info(f"{TARGET_FOLDER = }")    
    logger.info(f"{CLIP_SHAPEFILE = }")    
    logger.info(f"{LOG_FILE = }")

    logger.info("Started Programm")
    clip_tiffs(SOURCE_FOLDER, TARGET_FOLDER, CLIP_SHAPEFILE)
    logger.info("Ended Program\n")