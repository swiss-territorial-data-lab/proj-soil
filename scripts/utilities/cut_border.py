import rasterio

import yaml
import os
import sys
from glob import glob

import argparse

from loguru import logger
from tqdm import tqdm


def cut_border(SOURCE_FOLDER, TARGET_FOLDER, B) -> None:
    
    """
    Remove few border pixels, as much as given in parameter, of each input image

    Args:
        SOURCE_FOLDER (str): Path to the folder containing the original raster files.
        TARGET_FOLDER (str): Path to the folder where the cropped raster files will be saved.

    Returns:
        None
    """

    file_list = [os.path.relpath(x, start=SOURCE_FOLDER) for x in glob(f"{SOURCE_FOLDER}/**/*.tif", recursive=True) + glob(f"{SOURCE_FOLDER}/**/*.tiff", recursive=True)]
    for _, file in tqdm(enumerate(file_list), total=len(file_list), desc='Processing tiles'):

        source_file_path = os.path.join(SOURCE_FOLDER, file)
        with rasterio.open(source_file_path) as src:
            original = src.read()
            srcprof = src.profile.copy()

        L = src.shape[1]-2*B
        cropped = original[0:3,B:B+L,B:B+L]
        p1 = srcprof['transform'][0]
        p2 = srcprof['transform'][1]
        lat = srcprof['transform'][2]
        p4 = srcprof['transform'][3]
        p5 = srcprof['transform'][4]
        long = srcprof['transform'][5]
        RES = abs(p1)
        
        srcprof.update({
            'width': L,
            'height': L,
            'transform': (p1, p2, lat+B*RES, p4, p5, long-B*RES)
        })

        out_filename = os.path.join(TARGET_FOLDER, file)
        os.makedirs(os.path.dirname(out_filename), exist_ok=True)
        with rasterio.open(out_filename , "w", **srcprof) as tgt:
            tgt.write(cropped)




if __name__ == "__main__":
    # Argument and parameter specification
    parser = argparse.ArgumentParser(
        description="The script removes few border pixels of each input image.")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        default="/proj-soils/config/config-utilities.yaml")
    args = parser.parse_args()

    # load input parameters
    with open(args.config_file) as fp:
        cfg = yaml.load(
            fp,
            Loader=yaml.FullLoader)[os.path.basename(__file__)]

    SOURCE_FOLDER = cfg["source_folder"]
    TARGET_FOLDER = cfg["target_folder"]
    RES = cfg["resolution"]
    B = cfg["border_width"]
    L = cfg["length_center"]
    LOG_FILE = cfg["log_file"]

    # set up logger
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(LOG_FILE, level="INFO")

    logger.info(f"{SOURCE_FOLDER = }")
    logger.info(f"{TARGET_FOLDER = }")     
    logger.info(f"{B = }")      
    logger.info(f"{LOG_FILE = }")

    logger.info("Started Programm")
    cut_border(SOURCE_FOLDER, TARGET_FOLDER, B)
    logger.info("Ended Program/n")