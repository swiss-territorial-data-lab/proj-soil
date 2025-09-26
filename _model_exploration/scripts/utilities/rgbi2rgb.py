

import yaml
import os
import sys

import argparse

from loguru import logger

import rasterio

import numpy as np


def rgbi2rgb(SOURCE_FOLDER, TARGET_FOLDER) -> None:
    
    """
    Convert RGBI (Red, Green, Blue, Infrared) raster files to RGB (Red, Green, Blue) raster files.

    Args:
        SOURCE_FOLDER (str): Path to the folder containing the RGBI raster files.
        TARGET_FOLDER (str): Path to the folder where the converted RGB raster files will be saved.

    Returns:
        None
    """

    for file in os.listdir(SOURCE_FOLDER):
        if not file.endswith((".tif", ".tiff")):
            continue
        
        logger.info(f"Processing {file}")

        # Open the source raster file
        source_file_path = os.path.join(SOURCE_FOLDER, file)
        with rasterio.open(source_file_path) as src:
            rgbi = src.read()
            # Read the file profile
            srcprof = src.profile.copy()

        rgb = rgbi[1:4]
        
        # Create and save the RGB raster file
        with rasterio.open(os.path.join(TARGET_FOLDER, file), "w", **srcprof) as tgt:
            tgt.write(rgb)




if __name__ == "__main__":
    # Argument and parameter specification
    parser = argparse.ArgumentParser(
        description="The script converts all the tiff files in a given directory from RGBI to RGB")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        default="/Users/nicibe/Desktop/Job/swisstopo_stdl/soil_fribourg/proj-soils/config/config-infere_heig-vd.yaml")
    args = parser.parse_args()

    # load input parameters
    with open(args.config_file) as fp:
        cfg = yaml.load(
            fp,
            Loader=yaml.FullLoader)[os.path.basename(__file__)]

    SOURCE_FOLDER = cfg["source_folder"]
    TARGET_FOLDER = cfg["target_folder"]
    LOG_FILE = cfg["log_file"]

    # set up logger
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")
    logger.add(LOG_FILE, level="INFO")

    logger.info(f"{SOURCE_FOLDER = }")
    logger.info(f"{TARGET_FOLDER = }")      
    logger.info(f"{LOG_FILE = }")

    os.makedirs(TARGET_FOLDER, exist_ok=True)

    logger.info("Started Programm")
    rgbi2rgb(SOURCE_FOLDER, TARGET_FOLDER)
    logger.info("Ended Program\n")