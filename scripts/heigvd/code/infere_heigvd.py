# Copyright (c) OpenMMLab. All rights reserved.

import warnings
warnings.simplefilter("ignore", UserWarning)

from loguru import logger
import argparse
import yaml

import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,50))
import mmcv
import numpy as np

import rasterio

import sys

import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv.runner import load_checkpoint
from mmseg.core import get_classes
import cv2
import os.path as osp
import torch
import time
from tqdm import tqdm

import scripts.utilities.constants as cst



def infere_heigvd(MODEL_CONFIG, CHECKPOINT, SOURCE_FOLDER, TARGET_FOLDER, DEVICE, STRIDE, SIDE_LENGTH, PALETTE, logger):
    """
    Perform inference on images using the specified configuration and model checkpoint.

    Args:
        MODEL_CONFIG (str): Path to the configuration file.
        CHECKPOINT (str): Path to the model checkpoint.
        SOURCE_FOLDER (str or list): Path(s) to the folder(s) containing the input images.
        TARGET_FOLDER (str or list): Path(s) to the folder(s) where the output images will be saved.
        DEVICE (str): Device to run the inference on.
        STRIDE (int or list): Stride(s) to use for the inference.
        SIDE_LENGTH (int or list): Input resolution(s) for the images.
        PALETTE (list): List of colors for visualizing the segmentation results.
        logger: Logger object for logging messages.

    Returns:
        None
    """


    # Convert single string inputs to lists
    if isinstance(SOURCE_FOLDER, str):
        SOURCE_FOLDER = [SOURCE_FOLDER]
    if isinstance(TARGET_FOLDER, str):
        TARGET_FOLDER = [TARGET_FOLDER]
    if isinstance(STRIDE, int):
        STRIDE = [STRIDE]
    # if isinstance(SIDE_LENGTH, int):
    #     SIDE_LENGTH = [SIDE_LENGTH]

    assert len(SOURCE_FOLDER) == len(TARGET_FOLDER) == len(STRIDE) == len(SIDE_LENGTH)

    # Load the configuration file and initialize the segmentation model
    config_mmcv = mmcv.Config.fromfile(MODEL_CONFIG)


    model = init_segmentor(config_mmcv, checkpoint=None, device=DEVICE)
    checkpoint = load_checkpoint(model, CHECKPOINT, map_location='cpu')

    # Set the classes based on the checkpoint metadata or the provided palette
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = get_classes(PALETTE)

    # Iterate over the input folders and perform inference
    for source_folder, target_folder, stride, side_length in zip(SOURCE_FOLDER, TARGET_FOLDER, STRIDE, SIDE_LENGTH):
        logger.info("#########################")
        logger.info(f"Current configuration: {source_folder = }, {target_folder = }, {stride = }, {side_length = }")

        # Create the target folder if it doesn't already exist
        target_folder_pred = os.path.join(target_folder, 'pred')
        mmcv.mkdir_or_exist(target_folder_pred)
        if cst.LOGIT:
            target_folder_score_diff = os.path.join(target_folder, 'score_diff')
            mmcv.mkdir_or_exist(target_folder_score_diff)

        # Overwrite the model configuration to use the specified stride and input resolution
        config_mmcv["model"]["test_cfg"]["stride"] = (stride, stride)
        config_mmcv["test_pipeline"][1]["img_scale"] = tuple(side_length)
        config_mmcv["data"]["test"]["pipeline"][1]["img_scale"] = tuple(side_length)

        # Get the list of image files in the source folder
        list_image = os.listdir(source_folder)
        len_tot = len(list_image)
        start = time.time()

        # Process each image in the source folder
        for i, image_name in tqdm(enumerate(list_image), total=len(list_image), desc='Processing tiles'):


            # Skip non-TIFF files
            if not image_name.endswith(("tif", "tiff")):
                continue

            file_out_path = os.path.join(target_folder_pred, image_name)
            if os.path.exists(file_out_path) and not cst.OVERWRITE:
                logger.info(f"Already processed file {image_name}, continuing with next file")
                continue

            file_path_img = os.path.join(source_folder, image_name)

            # Read the image and convert it to RGB
            img = cv2.imread(file_path_img, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Open the image using rasterio to get the profile
            with rasterio.open(file_path_img, "r") as src:
                profile = src.profile
                if "photometric" in profile:
                    del profile["photometric"]

                profile.update(count=1, compress='lzw', dtype='uint8') 

                if cst.LOGIT:
                    profile_logit = profile.copy()
                    profile_logit.update(dtype='float32')
                    file_out_path_logit = os.path.join(target_folder_score_diff, image_name)

            height, width, _ = img.shape

            # Perform inference on the image
            if cst.LOGIT:
                result, logit = inference_segmentor(model, img)
            else:
                result = inference_segmentor(model, img)
           
            del img

            # release cuda memory
            torch.cuda.empty_cache()

            # Save the segmentation result as a new image
            with rasterio.open(file_out_path, 'w', **profile) as dst:
                dst.write(np.asarray(result, np.uint8).reshape((height, width)), 1)
            if cst.LOGIT:
                with rasterio.open(file_out_path_logit, 'w', **profile_logit) as dst:
                   dst.write(logit[0])           
    del model, checkpoint
    

if __name__ == "__main__":
    # Argument and parameter specification
    parser = argparse.ArgumentParser(
        description="The script rasterizes a given polygon")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        default="/proj-soils/config/config-pp.yaml")
    args = parser.parse_args()  

    # load input parameters
    with open(args.config_file) as fp:
        cfg = yaml.load(
            fp,
            Loader=yaml.FullLoader)[os.path.basename(__file__)]

    MODEL_CONFIG = cfg["model_config"]
    CHECKPOINT = cfg["checkpoint"]
    SOURCE_FOLDER = cfg["source_folder"]
    TARGET_FOLDER = cfg["target_folder"]
    DEVICE = cfg["device"]
    STRIDE = cfg["stride"]
    SIDE_LENGTH = cfg["side_length"]
    PALETTE = cfg["palette"]
    LOG_FILE = cfg["log_file"]


    # set up logger
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")
    logger.add(LOG_FILE, level="INFO")

    logger.info(f"{MODEL_CONFIG = }")
    logger.info(f"{CHECKPOINT = }")
    logger.info(f"{SOURCE_FOLDER = }")
    logger.info(f"{TARGET_FOLDER = }")
    logger.info(f"{DEVICE = }")
    logger.info(f"{STRIDE = }")
    logger.info(f"{SIDE_LENGTH = }")
    logger.info(f"{PALETTE = }")
    logger.info(f"{LOG_FILE = }")


    logger.info("Started Program")
    global_start_time = time.time()
    infere_heigvd(MODEL_CONFIG, CHECKPOINT, SOURCE_FOLDER, TARGET_FOLDER, DEVICE, STRIDE, SIDE_LENGTH, PALETTE, logger)
    global_stop_time = time.time()
    logger.info(f"Ended Program, time elapsed: {round(global_stop_time - global_start_time)} seconds")