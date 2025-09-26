import numpy as np
import yaml
import os
import sys
import shutil
from pathlib import Path
import argparse
import time

from loguru import logger


def random_split(SOURCE_IPT_FOLDER, SOURCE_TGT_FOLDER, TARGET_ROOT, SEED, SPLIT_FILE=None) -> None:
    """
    Randomly splits the raster files in the source folders into train, validation, and test sets.
    Copies the files to the corresponding folders in the target root directory.
    Two modes:
        (1) if a YAML file of the split to apply is given (SPLIT_FILE), it is directly used to split
        the tiles in the correct dataset
        (2) if it is not the case, SEED is used to create a random split and a split.yaml file 
        that contains the mapping of file IDs to their respective splits is generated
    
    Parameters:
        SOURCE_IPT_FOLDER (str): Path to the source folder containing input raster files.
        SOURCE_TGT_FOLDER (str): Path to the source folder containing target raster files.
        TARGET_ROOT (str): Path to the target root directory to save the split files.
        SEED (int): Seed value for random number generation.
        SPLIT_FILE (str): Path to the split file. If None, a new split file is generated.
    
    Returns:
        None

    Example:
        source_ipt_folder = 'input_rasters/'
        source_tgt_folder = 'target_rasters/'
        target_root = 'split_data/'
        seed = 42
        random_split(source_ipt_folder, source_tgt_folder, target_root, seed)
    """
    
    Path(TARGET_ROOT).mkdir(parents=True, exist_ok=True)


    ipt_ids = []
    tgt_ids = []

    ipt_files = list(os.listdir(SOURCE_IPT_FOLDER))
    for file in ipt_files:
        if not file.endswith(".tif"):
            continue
        ipt_id = file.split("-")[2].rstrip(".tif")
        ipt_ids.append(ipt_id)

    tgt_files = list(os.listdir(SOURCE_TGT_FOLDER))
    for file in tgt_files:
        if not file.endswith(".tif"):
            continue
        tgt_id = file.split("-")[2].rstrip(".tif")
        tgt_ids.append(tgt_id)

    ipt_ids = set(ipt_ids)
    tgt_ids = set(tgt_ids)
    ipt_ids = ipt_ids.intersection(tgt_ids)
    ids = list(ipt_ids)
    ids.sort()

    if SPLIT_FILE is None:
        np.random.seed(SEED)
        np.random.shuffle(ids)

        n = len(ids)
        n_train = int(n * 0.7)
        n_val = n_train + int(np.floor(n - n_train)/2)
        train_ids = ids[:n_train]
        val_ids = ids[n_train:n_val]
        test_ids = ids[n_val:]

        split_dic = {"train": sorted(train_ids), "val": sorted(val_ids), "test":sorted(test_ids)}
        with open(f'{TARGET_ROOT}/split.yaml', 'w') as file:
            yaml.dump(split_dic, file)
        logger.info(f"Generated new random split with seed {SEED} and saved to {TARGET_ROOT}/split.yaml")

    else:
        with open(SPLIT_FILE, 'r') as file:
            split_dic = yaml.load(file, Loader=yaml.FullLoader)

        shutil.copyfile(SPLIT_FILE, f"{TARGET_ROOT}/split.yaml")

        train_ids = split_dic["train"]
        val_ids = split_dic["val"]
        test_ids = split_dic["test"]
        
        logger.info(f"Loaded split from {SPLIT_FILE}")


    for split in ["train", "val","test"]:
        Path(f"{TARGET_ROOT}/{split}/tgt").mkdir(parents=True, exist_ok=True)
        Path(f"{TARGET_ROOT}/{split}/ipt").mkdir(parents=True, exist_ok=True)

    traincount = 0
    valcount = 0
    testcount = 0
    for file in sorted(tgt_files):
        if not file.endswith(".tif"):
            continue
        if not file in ipt_files:
            print(f"{file} in ipt_files, but not in tgt_files\n")
            continue

        print(f'{file = }', end="\r")
        id = int(file.split("-")[2].rstrip(".tif"))
        if id in train_ids:
            shutil.copyfile(f"{SOURCE_IPT_FOLDER}/{file}", f"{TARGET_ROOT}/train/ipt/{file}")
            shutil.copyfile(f"{SOURCE_TGT_FOLDER}/{file}", f"{TARGET_ROOT}/train/tgt/{file}")
            traincount += 1
        elif id in val_ids:
            shutil.copyfile(f"{SOURCE_IPT_FOLDER}/{file}", f"{TARGET_ROOT}/val/ipt/{file}")
            shutil.copyfile(f"{SOURCE_TGT_FOLDER}/{file}", f"{TARGET_ROOT}/val/tgt/{file}")
            valcount += 1
        elif id in test_ids:
            shutil.copyfile(f"{SOURCE_IPT_FOLDER}/{file}", f"{TARGET_ROOT}/test/ipt/{file}")
            shutil.copyfile(f"{SOURCE_TGT_FOLDER}/{file}", f"{TARGET_ROOT}/test/tgt/{file}")
            testcount += 1
    logger.info(f"{traincount = }")
    logger.info(f"{valcount = }")
    logger.info(f"{testcount = }")
    

if __name__ == "__main__":
    
    # Argument and parameter specification
    parser = argparse.ArgumentParser(
        description="The script rasterizes a given polygon")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        default="/proj-soils/config/config-utilities.yaml"
        )
    args = parser.parse_args()

    # load input parameters
    with open(args.config_file) as fp:
        cfg = yaml.load(
            fp,
            Loader=yaml.FullLoader)[os.path.basename(__file__)]


    SOURCE_IPT_FOLDER = cfg["source_ipt_folder"]
    SOURCE_TGT_FOLDER = cfg["source_tgt_folder"]
    TARGET_ROOT = cfg["target_root"]
    SEED = cfg["seed"]
    SPLIT_FILE = cfg["split_file"]
    LOG_FILE = cfg["log_file"]

    # set up logger
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")
    logger.add(LOG_FILE, level="INFO")

    logger.info(f"{SOURCE_IPT_FOLDER = }")
    logger.info(f"{SOURCE_TGT_FOLDER = }")
    logger.info(f"{TARGET_ROOT = }")    
    logger.info(f"{SEED = }")
    logger.info(f"{SPLIT_FILE = }")
    logger.info(f"{LOG_FILE = }")    

    os.makedirs(TARGET_ROOT, exist_ok=True)

    logger.info("Started Programm")
    random_split(SOURCE_IPT_FOLDER, SOURCE_TGT_FOLDER, TARGET_ROOT, SEED, SPLIT_FILE)
    logger.info("Ended Program\n")
