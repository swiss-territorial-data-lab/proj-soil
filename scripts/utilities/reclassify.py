import rasterio

import numpy as np

import yaml
import os
import sys

import argparse

from loguru import logger


def reclassify(SOURCE_FOLDER, TARGET_FOLDER, MAPPING) -> None:
    """
    Reclassify raster files in the source folder and save the reclassified
    files to the target folder based on the provided mapping.
    
    Parameters:
        SOURCE_FOLDER (str): Path to the source folder containing raster files.
        TARGET_FOLDER (str): Path to the target folder to save reclassified raster files.
        MAPPING (dict): Dictionary mapping original values to new reclassified values.
    
    Returns:
        None

    Example:
        source_folder = 'original_rasters/'
        target_folder = 'reclassified_rasters/'
        mapping = {0: 1, 1: 2, 2: 0}  # Example mapping: 0->1, 1->2, 2->0

        reclassify_raster_files(source_folder, target_folder, mapping)
    """
    for file in os.listdir(SOURCE_FOLDER):
        if not file.endswith((".tif", ".tiff")):
            continue
        
        logger.info(f"Reclassifying {file}")

        # Open the source raster file
        source_file_path = os.path.join(SOURCE_FOLDER, file)
        with rasterio.open(source_file_path) as src:
            ar = src.read(1)
            # Read the file profile
            srcprof = src.profile.copy()

        # Reclassify the array based on the provided mapping
        ar = np.vectorize(MAPPING.__getitem__)(ar)
        
        # Create and save the reclassified raster file
        with rasterio.open(os.path.join(TARGET_FOLDER, file), "w", **srcprof) as tgt:
            tgt.write(ar, indexes=1)




if __name__ == "__main__":
    # Argument and parameter specification
    parser = argparse.ArgumentParser(
        description="The script rasterizes a given polygon")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        default="/Users/nicibe/Desktop/Job/swisstopo_stdl/soil_fribourg/proj-soils/config/config-eval_gt.yaml")
    args = parser.parse_args()

    # load input parameters
    with open(args.config_file) as fp:
        cfg = yaml.load(
            fp,
            Loader=yaml.FullLoader)[os.path.basename(__file__)]

    SOURCE_FOLDER = cfg["source_folder"]
    TARGET_FOLDER = cfg["target_folder"]
    MAPPING = cfg["mapping"]
    LOG_FILE = cfg["log_file"]

    # set up logger
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")
    logger.add(LOG_FILE, level="INFO")

    logger.info(f"{SOURCE_FOLDER = }")
    logger.info(f"{TARGET_FOLDER = }")    
    logger.info(f"{MAPPING = }")    
    logger.info(f"{LOG_FILE = }")

    logger.info("Started Programm")
    reclassify(SOURCE_FOLDER, TARGET_FOLDER, MAPPING)
    logger.info("Ended Program\n")