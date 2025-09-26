
import yaml
import os
import sys

import argparse

import rasterio
from rasterio.enums import ColorInterp
from rasterio.io import MemoryFile
import numpy as np

from loguru import logger
from tqdm import tqdm


colormap = {
    0: (0, 0, 0),     
    1: (204, 204, 204),         
    2: (148, 91, 27),         
    3: (0, 184, 255),       
    4: (255, 255, 255),  
}

colormap = {
    0: (0, 0, 0),     
    1: (73, 73, 231),         
    2: (173, 148, 82),         
    3: (240, 77, 104),         
    4: (146, 146, 146),
    5: (0, 184, 255),
    6: (0, 145, 147),
    7: (255, 255, 255),
    8: (69, 183, 47),
    9: (91, 192, 140),
    10: (141, 186, 55),
    11: (60, 60, 60),
    12: (255, 193, 0),
    13: (0, 253, 246),
    14: (255, 255, 0),
    15: (0, 184, 254),       
    16: (0, 255, 139),        
}


def to_colormap(SOURCE_FOLDER, TARGET_FOLDER, B, L, RES) -> None:
    
    """
    Convert rasters in a folder to color map rasters.

    Args:
        SOURCE_FOLDER (str): Path to the folder containing the original raster files.
        TARGET_FOLDER (str): Path to the folder where the coror map raster files will be saved.

    Returns:
        None
    """

    file_list = os.listdir(SOURCE_FOLDER)
    os.makedirs(TARGET_FOLDER, exist_ok=True)

    for _,file in tqdm(enumerate(file_list),total=len(file_list), desc='Processing tiles'):
        if not file.endswith((".tif", ".tiff")):
            continue

        source_file_path = os.path.join(SOURCE_FOLDER, file)
        with rasterio.open(source_file_path) as src:
            original = src.read()
            srcprof = src.profile.copy()

        # Save to GeoTIFF with colormap
        with rasterio.open(
            os.path.join(TARGET_FOLDER, file),
            'w',
            driver='GTiff',
            height=original.shape[0],
            width=original.shape[1],
            count=1,
            dtype='uint8',
            crs='EPSG:2056',
            transform=rasterio.transform.from_origin(0, 0, 1, 1)
            # **scrprof)
        ) as dst:
            dst.write(original, 1)
            dst.write_colormap(1, colormap)
            dst.colorinterp = [ColorInterp.palette]


if __name__ == "__main__":
    # Argument and parameter specification
    parser = argparse.ArgumentParser(
        description="The script converts rasters with 4 categories into color map rasters.")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        default="C:/Users/cmarmy/Documents/STDL/Sols/proj-soils/config/config-utilities.yaml")
    args = parser.parse_args()

    # load input parameters
    with open(args.config_file) as fp:
        cfg = yaml.load(
            fp,
            Loader=yaml.FullLoader)[os.path.basename(__file__)]

    SOURCE_FOLDER = cfg["source_folder"]
    TARGET_FOLDER = cfg["target_folder"]
    COLOR_MAP = cfg["color_map"]
    LOG_FILE = cfg["log_file"]

    # set up logger
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(LOG_FILE, level="INFO")

    logger.info(f"{SOURCE_FOLDER = }")
    logger.info(f"{TARGET_FOLDER = }")     
    logger.info(f"{COLOR_MAP = }")
    logger.info(f"{LOG_FILE = }")

    logger.info("Started Programm")
    to_colormap(SOURCE_FOLDER, TARGET_FOLDER, COLOR_MAP)
    logger.info("Ended Program/n")