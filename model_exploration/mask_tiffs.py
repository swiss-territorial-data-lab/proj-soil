import rasterio
from rasterio.mask import mask

import geopandas as gpd

import numpy as np

import yaml
import os
import sys

import argparse

from loguru import logger


def mask_tiffs(SOURCE_FOLDER, TARGET_FOLDER, MASK, NODATA) -> None:
    """
    Apply a mask to tiff files in the specified folder. Pixels outside
    the specified mask vector layer are reclassified as the specified
    nodata value.
    
    Parameters:
        SOURCE_FOLDER (str): Path to the source folder containing raster files.
        TARGET_FOLDER (str): Path to the target folder to save the masked raster files.
        MASK (str): Path to a polygon vector file to use as mask.
    
    Returns:
        None

    Example:
        source_folder = 'original_rasters/'
        target_folder = 'reclassified_rasters/'
        mask = 'mask.gpkg'
        reclassify_raster_files(source_folder, target_folder, mask)
    """

    mask_gdf = gpd.read_file(MASK)
    for file in os.listdir(SOURCE_FOLDER):
        if not file.endswith(".tif"):
            continue
        
        logger.info(f"Masking {file}")
        

        # Open the source raster file
        source_file_path = os.path.join(SOURCE_FOLDER, file)
        with rasterio.open(source_file_path) as src:
            # Read the file profile
            srcprof = src.profile.copy()
            
            # Apply the mask to the raster file, reclassifying pixels
            # outside the mask to the specified nodata value
            ar_masked, _ = mask(src, mask_gdf.geometry, nodata=NODATA)
        
        srcprof["nodata"] = NODATA

        # Create and save the reclassified raster file
        with rasterio.open(os.path.join(TARGET_FOLDER, file), "w", **srcprof) as tgt:
            tgt.write(ar_masked)


if __name__ == "__main__":
    # Argument and parameter specification
    parser = argparse.ArgumentParser(
        description="The script rasterizes a given polygon")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        # default="proj-soils/config/config-eval_heig-vd.yaml"
        default="proj-soils/config/config-eval_IGN.yaml"
        # default="proj-soils/config/config-eval_OFS.yaml"
        )
    args = parser.parse_args()

    # load input parameters
    with open(args.config_file) as fp:
        cfg = yaml.load(
            fp,
            Loader=yaml.FullLoader)[os.path.basename(__file__)]

    SOURCE_FOLDER = cfg["source_folder"]
    TARGET_FOLDER = cfg["target_folder"]
    MASK = cfg["mask"]
    NODATA = cfg["nodata"]
    LOG_FILE = cfg["log_file"]

    # set up logger
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")
    logger.add(LOG_FILE, level="INFO")

    logger.info(f"{SOURCE_FOLDER = }")
    logger.info(f"{TARGET_FOLDER = }")    
    logger.info(f"{MASK = }")    
    logger.info(f"{NODATA = }")    
    logger.info(f"{LOG_FILE = }")

    logger.info("Started Programm")
    mask_tiffs(SOURCE_FOLDER, TARGET_FOLDER, MASK, NODATA)
    logger.info("Ended Program\n")