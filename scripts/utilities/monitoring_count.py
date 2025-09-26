import sys
import os
import yaml
import argparse
import time
from loguru import logger
from tqdm import tqdm
from collections import defaultdict

import rasterio
import rasterio.mask
import fiona
import numpy as np

def main(DIR_CLS1, dict_tag, MASK=None, invert_mask=False):
    '''
    This script counts the number of pixels by type of change documented in a dictionnary.

    Parameters:
        DIR_CLS1 (str): folder with of rasters with tagged changes
        dict_tag (dict): pixel values with corresponding change tag
        MASK (str): path of vector data to mask the raster with. 
        invert_mask (bool): is the mask to be inverted before masking?
    
    Returns:
        prompt count per key of the dict_tag
    '''

    tag_list = []
    count_list = []
    if MASK is not None:
        with fiona.open(MASK, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]
    else:
        shapes = None

    file_list = os.listdir(DIR_CLS1)
    for _,file in tqdm(enumerate(file_list),total=len(file_list), desc='Processing tiles'):
        if not file.endswith((".tif", ".tiff")):
            continue

        source_file_path = os.path.join(DIR_CLS1, file)
        with rasterio.open(source_file_path) as src:
            if shapes:
                raster, _ = rasterio.mask.mask(src, shapes,invert=invert_mask)
            else:
                raster = src.read()
        
        # Count pixel values
        tag, counts = np.unique(raster, return_counts=True)
        tag_list.extend(list(tag))
        count_list.extend(list(counts))

    value_counts = defaultdict(int)

    for k, v in zip(tag_list, count_list):
        value_counts[k] = v + value_counts[k] 
    sum_counts = np.sum(count_list)
    sum_soil = 0
    sum_non_soil = 0

    print("{: <35} {: >10} {: >10}".format(*['TAG', '%', 'PX COUNT']))
    sorted_list = sorted(value_counts.items())
    for value, count in sorted_list:
        print("{: <35} {: >10} {: >10}".format(*[dict_tag[value], np.round(count/sum_counts*100,2), count]))
        if value in [20,21,22,23,23,25,26]:
            sum_non_soil = sum_non_soil + count
        if value in [30,31, 32, 33, 34, 35, 36]:
            sum_soil = sum_soil + count
    print("")
    print("{: <35} {: >10} {: >10}".format(*['Tot soil to non-soil', np.round(sum_non_soil/sum_counts*100,2), sum_non_soil]))
    print("{: <35} {: >10} {: >10}".format(*['Tot non-soil to soil', np.round(sum_soil/sum_counts*100,2), sum_soil]))


if __name__ == "__main__":

    # Argument and parameter specification
    parser = argparse.ArgumentParser(
        description="Compute count per class of a raster and print the results. ")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        default="/proj-soils/config/config-utilities.yaml")
    args = parser.parse_args()

    # load input parameters
    with open(args.config_file) as fp:
        cfg = yaml.load(
            fp,
            Loader=yaml.FullLoader)[os.path.basename(__file__)]

    dict_tag = cfg["dict_tag"]
    DIR_CLS = cfg["input_folder"]
    MASK = cfg["mask"]
    LOG_FILE = cfg["log_file"]

    # set up logger
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(LOG_FILE, level="INFO")

    logger.info(f"{DIR_CLS = }")     
    logger.info(f"{MASK = }")
    logger.info(f"{dict_tag = }")      
    logger.info

    tic = time.time()
    logger.info("Started Programm")

    main(DIR_CLS, dict_tag, MASK=None)
    
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")
