import os, sys
import argparse
import geopandas as gpd
import pandas as pd
import numpy as np
import yaml
import csv
import rasterio 

from loguru import logger
import time

def count_pixels(DATASET_DIR, SUB_DIRS, count_dict, CLASS_NAME):
    """
    Count pixels per class for rasters by folder (datasets) within a folder.  

    Parameters:
        DATASET_DIR (str): path where are the datasets (val, train, test)
        SUB_DIRS (list): list of subdirs name of the datasets on which count the pixels 
        count_dict (dict): initialisation of the dictionnary of counted pixels per class
        CLASS_NAME: names of classes corresponding to the count_dict. 
    
    Returns:
        CSV file with pixel counts per SUB_DIRS, saved in the DATASET_DIR directory
    """

    with open(os.path.join(DATASET_DIR,'pixel_counts.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write header row (keys)
        writer.writerow(CLASS_NAME)
            
    for dataset in SUB_DIRS:
        list_tiles = os.listdir(os.path.join(DATASET_DIR,dataset+'/tgt'))
        im_dir = os.path.join(DATASET_DIR,dataset+'/tgt')
        pixel_count_dict_dataset = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 
                                    10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0}
        pixel_count_dict_dataset_perc = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 
                                    10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0}

        for tile in list_tiles:

            if not tile.endswith('.tif'):
                continue

            # Open the raster file
            with rasterio.open(os.path.join(im_dir, tile)) as src:
                band = src.read(1)  

            nodata = src.nodata
            if nodata is not None:
                band = band[band != nodata]

            # Count unique values and their frequencies
            unique, counts = np.unique(band, return_counts=True)
            pixel_count_dict = dict(zip(unique, counts))
            # Print results
            for value, count in pixel_count_dict.items():
                count_dict[value]= count_dict[value]+pixel_count_dict[value]
                pixel_count_dict_dataset[value]= pixel_count_dict_dataset[value]+pixel_count_dict[value]

        sum_dataset = sum(pixel_count_dict_dataset.values())
        for value, count in pixel_count_dict_dataset.items():
            pixel_count_dict_dataset_perc[value]= np.round(count/sum_dataset*100,2)
        row = list(pixel_count_dict_dataset_perc.values())
        row.append(dataset)


        with open(os.path.join(DATASET_DIR,'pixel_counts.csv'), mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

    sum_gt = sum(count_dict.values())
    for value, count in count_dict.items():
        count_dict[value]= np.round(count/sum_gt*100,2)

    row = list(count_dict.values())
    row.append('gt')

    with open(os.path.join(DATASET_DIR,'pixel_counts.csv'), mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)


if __name__ == "__main__":
    # Argument and parameter specification
    parser = argparse.ArgumentParser(
        description="The script...")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        default="/proj-soils/config/config-utilities.yaml")
    args = parser.parse_args()

    # load input parameters
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]
    
    DATASET_DIR = cfg["dataset_dir"]
    SUB_DIRS = cfg["sub_dirs"]
    count_dict = cfg["pixel_count_dict"]
    CLASS_NAME = cfg["class_name"]   
    LOG_FILE = cfg["log_file"] 

    # set up logger
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")
    logger.add(LOG_FILE, level="INFO")

    logger.info(f"{DATASET_DIR = }")
    logger.info(f"{SUB_DIRS = }")
    logger.info(f"{count_dict = }")
    logger.info(f"{CLASS_NAME = }")


    tic =time.time()


    logger.info("Started Programm")
    count_pixels(DATASET_DIR, SUB_DIRS, count_dict, CLASS_NAME)
    logger.info(f"Ended Program {time.time()-tic} s\n")