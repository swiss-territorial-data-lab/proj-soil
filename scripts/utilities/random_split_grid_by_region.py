import os, sys
import yaml
import argparse

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio 

from loguru import logger
import time


def split_grid(SEED, GRID_PATH, CT_AOI, GRID_PATH_SPLIT, PREVIOUS_SPLIT, TARGET_ROOT, AOI_FILTER):
    """
    Randomly split the cells of the grid vector in train, validation and test sets 
    for the "sub_aoi" attribute by adding a new attribute called "split". 
    The random split of some sub-AOI is corrected for a previous split (see l. 51).
    The use of cantonal border allows the creation of different datasets:
         - agnostic of the canton
         - per canton
    YAML files corresponding to the different splits are saved. 
         

    Parameters:
        SEED (int): random seed for random sampling
        GRID_PATH (str): path to the grid of input tiles 
        CT_AOI (str): path to the vector file of cantonal borders
        GRID_PATH_SPLIT (str): path where to save the updated grid of input tiles 
        PREVOUS_SPLIT (str): path of a grid containing a previous split to reuse
        TARGET_ROOT (str): path where to save the dataset folders 
        AOI_FILTER (dict): key-value dict of the canton name in CT_AOI and name to use afterwards. 
    
    Returns:
        YAML files corresponding to the different splits
    """

    grid = gpd.read_file(GRID_PATH)
    grid_out = gpd.GeoDataFrame()

    for sub_aoi in set(list(grid['sub_aoi'])):
        sub_grid = grid[grid['sub_aoi']==sub_aoi]
        sub_grid = sub_grid.sample(frac=1, random_state=SEED).reset_index(drop=True)
        n = len(sub_grid)
        sub_grid["split"] = pd.Series([''] * n, dtype="str")
        n_train = int(n * 0.7)
        n_val = n_train + int(np.floor((n - n_train)/2))

        sub_grid.loc[:n_train, "split"] = "train"
        sub_grid.loc[n_train:n_val, "split"] = "val"
        sub_grid.loc[n_val:, "split"] = "test"

        grid_out = pd.concat([grid_out, sub_grid])

    # Merge split Phase 1 to grid Phase 2

    grid_p1 = gpd.read_file(PREVIOUS_SPLIT)
    grid_p1 = grid_p1[grid_p1['depth']==0]

    # Assigning val samples to test dataset. 
    n_val = len(grid_p1[grid_p1['split']=='val'])
    grid_p1.sort_values("split", ascending=False, inplace=True, ignore_index=True)
    grid_p1.loc[:int(np.floor(n_val/2)), "split"] = "test"
    grid_p1.loc[:,'geometry']=grid_p1.buffer(-25)

    grid_p2_split_p1 = gpd.sjoin(grid_out, grid_p1[['split','geometry']], how='left', predicate='contains')
    grid_p2_split_p1['final_split'] = grid_p2_split_p1["split_right"].fillna(grid_p2_split_p1["split_left"])
    grid_p2_split_p1.drop_duplicates(subset=['unique_id'], inplace=True)
    grid_p2_split_p1.drop(columns=['index_right'], inplace=True)

    # add column for canton
    ct_boundaries = gpd.read_file(CT_AOI)
    grid_p2_split_p1 = gpd.sjoin(grid_p2_split_p1,ct_boundaries[['name','geometry']], how='left',predicate='within')

    grid_p2_split_p1.to_file(GRID_PATH_SPLIT)

    for key, value in AOI_FILTER.items(): 
        grid_to_yaml = grid_p2_split_p1[grid_p2_split_p1['name']!=key]
        split_dic = {"train": sorted(list(grid_to_yaml[grid_to_yaml['final_split']=='train']['unique_id'])), 
                    "val": sorted(list(grid_to_yaml[grid_to_yaml['final_split']=='val']['unique_id'])), 
                    "test":sorted(list(grid_to_yaml[grid_to_yaml['final_split']=='test']['unique_id']))}
        with open(f'{TARGET_ROOT}/split_{value}.yaml', 'w') as file:
            yaml.dump(split_dic, file)

    # area_dic = {"land": sorted(list(grid_p2_split_p1[grid_p2_split_p1['area_type']=='land']['unique_id'])), 
    #             "urban": sorted(list(grid_p2_split_p1[grid_p2_split_p1['area_type']=='urban']['unique_id']))}
    # with open(f'{TARGET_ROOT}/split_area_type.yaml', 'w') as file:
    #     yaml.dump(area_dic, file)

    # grid_new = gpd.read_file("/proj-soils/train2/recursive_grids_max51-2m_splits_edited_del_fr_new.gpkg")
    # grid_old = gpd.read_file("/proj-soils/train2/recursive_grids_max51-2m_splits_edited_del_fr_old.gpkg")
    # area_dic = {"new": sorted(list(grid_new['unique_id'])), "old": sorted(list(grid_old['unique_id']))}
    # with open(f'/proj-soils/train2/split_old_new.yaml', 'w') as file:
    # yaml.dump(area_dic, file)


    return


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

    SEED = cfg["seed"]
    GRID_PATH = cfg["grid_path"]
    CT_AOI = cfg["path_aoi_canton"]
    GRID_PATH_SPLIT = cfg["grid_path_split"]
    PREVIOUS_SPLIT = cfg["previous_split"]
    TARGET_ROOT = cfg["target_root"]
    AOI_FILTER = cfg["aoi_filter"]
    LOG_FILE = cfg["log_file"] 

    # set up logger
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")
    logger.add(LOG_FILE, level="INFO")

    logger.info(f"{SEED = }")
    logger.info(f"{GRID_PATH = }")
    logger.info(f"{CT_AOI = }")
    logger.info(f"{GRID_PATH_SPLIT = }")
    logger.info(f"{PREVIOUS_SPLIT = }")
    logger.info(f"{TARGET_ROOT = }")
    logger.info(f"{AOI_FILTER = }")

    os.makedirs(TARGET_ROOT, exist_ok=True)

    tic =time.time()

    logger.info("Started Programm")
    split_grid(SEED, GRID_PATH, CT_AOI, GRID_PATH_SPLIT, PREVIOUS_SPLIT, TARGET_ROOT, AOI_FILTER)
    logger.info(f"Ended Program {time.time()-tic} s\n")