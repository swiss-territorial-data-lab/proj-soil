

import yaml
import os
import sys

import argparse

from loguru import logger
from tqdm import tqdm

import rasterio
import geopandas as gpd
import numpy as np

import multiprocessing
from multiprocessing import Pool
from functools import partial



def convert_encoding(file, SOURCE_FOLDER, TARGET_FOLDER) -> None:
    
    """
    Convert RGBI (Red, Green, Blue, Infrared) raster files to RGB (Red, Green, Blue) raster files.

    Args:
        SOURCE_FOLDER (str): Path to the folder containing the RGBI raster files.
        TARGET_FOLDER (str): Path to the folder where the converted RGB raster files will be saved.

    Returns:
        None
    """

    # for file in os.listdir(SOURCE_FOLDER):
    if not file.endswith((".tif", ".tiff")):
        return

    # Open the source raster file
    source_file_path = os.path.join(SOURCE_FOLDER, file)
    with rasterio.open(source_file_path) as src:
        data = src.read(1).astype(np.float32)
        srcprof = src.profile.copy()


        # Initialize output
        converted = np.zeros_like(data, dtype=np.uint8)

        # Case 1: linear scaling 0–0.25 → 0–250
        mask = (data >= 0) & (data <= 0.25)
        converted[mask] = (data[mask] / 0.25 * 250).astype(np.uint8)

        # Case 2+: discrete bins
        bins = [
            (data > 0.25) & (data <= 0.3),
            (data > 0.3) & (data <= 0.4),
            (data > 0.4) & (data <= 0.6),
            (data > 0.6) & (data <= 0.8),
            (data > 0.8) & (data <= 1.0),
        ]
        values = [251, 252, 253, 254, 255]

        converted = np.where(bins[0], values[0], converted)
        for b, v in zip(bins[1:], values[1:]):
            converted = np.where(b, v, converted)

        # Update profile for int8
        srcprof.update(
            dtype=rasterio.uint8,
            count=1,
            compress="lzw"
        )

    
    # Create and save the RGB raster file
    with rasterio.open(os.path.join(TARGET_FOLDER, file), "w", **srcprof) as tgt:
        tgt.write(converted, 1)



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

    list_files = os.listdir(SOURCE_FOLDER)

    num_cores = multiprocessing.cpu_count()-1
    logger.info(f"Starting job on {num_cores} cores...") 
    partial_func = partial(convert_encoding,SOURCE_FOLDER=SOURCE_FOLDER, TARGET_FOLDER=TARGET_FOLDER)


    with Pool(num_cores, maxtasksperchild=100) as p:
        for _ in tqdm(p.imap(partial_func, list_files, chunksize=1), total=len(list_files)):
            pass
    logger.info("Ended Program\n")