import os
import sys

import yaml
import argparse

from loguru import logger
import time

import infere_heigvd as infere_heigvd # __init_ to do when importing    
import scripts.utilities.cut_border as cut_border
import scripts.utilities.post_processing_embedding as post_processing_embedding
import scripts.utilities.mosaic_grid as mosaic_grid
import scripts.utilities.fct_misc as misc
import scripts.utilities.constants as cst


if __name__ == "__main__":
    """
    With constants.py, allows to run successively different steps. 
    * cst.INFER=True: infers with infere_heigvd.py
    * cst.BORDER=True: cuts prediction tiles with cut_border.py
    * cst.CORRECT=True: corrects predictions tiles with post_processing_embedding.py
    * cst.MOSAIC=True: mosaics predictions tiles and corrected prediction tiles with mosaic.py

    Parameters:
        WORKING_DIR (str): working directory 

        ... for further parameters, go see the corresponding script. 
    
    Returns:
        Prompt the processing time for each task

        ... for further outputs, go see the corresponding script. 
    """


   # Argument and parameter specification
    parser = argparse.ArgumentParser(
        description="The script processes regions into soil cover map.")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        default="/proj-soils/config/config-pp.yaml")
    args = parser.parse_args()

    global_start_time = time.time()

    with open(args.config_file) as fp:
        cfg = yaml.load(
            fp,
            Loader=yaml.FullLoader)['productive_pipeline.py']

    WORKING_DIR = cfg["working_dir"]

    logger.info(f"{WORKING_DIR = }")

    os.chdir(WORKING_DIR)

    if cst.INFER:
        with open(args.config_file) as fp:
            cfg = yaml.load(
                fp,
                Loader=yaml.FullLoader)['infere_heigvd.py']

        MODEL_CONFIG = cfg["model_config"]
        CHECKPOINT = cfg["checkpoint"]
        ORTHO_PATCH_FOLDER = cfg["source_folder"]
        PRED_PATCH_FOLDER = cfg["target_folder"]
        DEVICE = cfg["device"]
        STRIDE = cfg["stride"]
        SIDE_LENGTH = cfg["side_length"]
        PALETTE = cfg["palette"]
        LOG_FILE = cfg["log_file"]

        logger = misc.format_logger(logger, LOG_FILE)
        logger.info("---------------------")
        logger.info(f"{MODEL_CONFIG = }")
        logger.info(f"{CHECKPOINT = }")
        logger.info(f"{ORTHO_PATCH_FOLDER = }")
        logger.info(f"{PRED_PATCH_FOLDER = }")
        logger.info(f"{DEVICE = }")
        logger.info(f"{STRIDE = }")
        logger.info(f"{SIDE_LENGTH = }")
        logger.info(f"{PALETTE = }")
        logger.info(f"{LOG_FILE = }")

        logger.info("Infering soil cover...")
        tic_infer = time.time()

        infere_heigvd.infere_heigvd(MODEL_CONFIG, CHECKPOINT, ORTHO_PATCH_FOLDER, PRED_PATCH_FOLDER, DEVICE, STRIDE, SIDE_LENGTH, PALETTE, logger)
        toc_infer = time.time()
        logger.info(f"...ended process after {toc_infer-tic_infer} s.\n")


    if cst.CUT_BORDER:
        with open(args.config_file) as fp:
            cfg = yaml.load(
                fp,
                Loader=yaml.FullLoader)['cut_border.py']

        PRED_PATCH_FOLDER = cfg["source_folder"]
        CUT_PATCH_FOLDER = cfg["target_folder"]
        B = cfg["border_width"]
        LOG_FILE = cfg["log_file"]

        logger = misc.format_logger(logger, LOG_FILE)
        logger.info("---------------------")
        logger.info(f"{PRED_PATCH_FOLDER = }")
        logger.info(f"{CUT_PATCH_FOLDER = }")     
        logger.info(f"{B = }")      
        logger.info(f"{LOG_FILE = }")

        logger.info("Cutting border...")
        tic_cut = time.time()

        cut_border.cut_border(os.path.join(PRED_PATCH_FOLDER,'pred'), os.path.join(CUT_PATCH_FOLDER,'pred'), B)
        toc_cut = time.time()
        logger.info(f"...ended process after {toc_cut-tic_cut} s.\n")

        PRED_PATCH_FOLDER_SCORE_DIFF = os.path.join(PRED_PATCH_FOLDER,'score_diff')
        if cst.LOGIT and os.path.exists(PRED_PATCH_FOLDER_SCORE_DIFF):
            logger.info("Cutting border of score_diff...")
            tic_cut = time.time()

            cut_border.cut_border(PRED_PATCH_FOLDER_SCORE_DIFF, os.path.join(CUT_PATCH_FOLDER,'score_diff'), B)
            toc_cut = time.time()
            logger.info(f"...ended process after {toc_cut-tic_cut} s.\n")

    if cst.CORRECT:
        # Correct the artefacts
        with open(args.config_file) as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)['post_processing_embedding.py']
        
        CORRECTION_TYPE = cfg["correction_type"]
        RECLASSIFICATION_RULES = cfg["reclassification_rules"]
        INPUT_FOLDER_1 = cfg["input_folder_1"]
        INPUT_FOLDER_2 = cfg["input_folder_2"]
        CONF_FOLDER_1 = cfg["conf_folder_1"]
        CONF_FOLDER_2 = cfg["conf_folder_2"]
        CORRECTED_FOLDER_1 = cfg["corrected_folder_1"]
        CORRECTED_FOLDER_2 = cfg["corrected_folder_2"]    
        MONITORED_FOLDER = cfg["monitored_folder"]
        LOG_FILE = cfg["log_file"]

        logger = misc.format_logger(logger, LOG_FILE)
        logger.info("---------------------")
        logger.info(f"{CORRECTION_TYPE = }")
        logger.info(f"{RECLASSIFICATION_RULES = }")      
        logger.info(f"{INPUT_FOLDER_1 = }") 
        logger.info(f"{INPUT_FOLDER_2 = }") 
        logger.info(f"{CONF_FOLDER_1 = }")
        logger.info(f"{CONF_FOLDER_2 = }")
        logger.info(f"{CORRECTED_FOLDER_1 = }")
        logger.info(f"{CORRECTED_FOLDER_2 = }")
        logger.info(f"{MONITORED_FOLDER = }") 

        logger.info("Correcting artefacts...")
        tic_arte = time.time()

        post_processing_embedding.post_processing_embedding(INPUT_FOLDER_1, INPUT_FOLDER_2, CONF_FOLDER_1, CONF_FOLDER_2, CORRECTED_FOLDER_1, CORRECTED_FOLDER_2, 
                              MONITORED_FOLDER, CORRECTION_TYPE, RECLASSIFICATION_RULES)
        toc_arte = time.time()
        logger.info(f"...ended process after {toc_arte-tic_arte} s.\n")

    if cst.MOSAIC:
        # Mosaic the tiles
        with open(args.config_file) as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)['mosaic_grid.py']
        
        LOG_FILE = cfg["log_file"]
        PATCH_FOLDER = cfg["input_infer_folder"]
        PATCH_CORR_FOLDER = cfg["input_corr_folder"]
        MOSAIC_FOLDER = cfg["output_folder"]
        CRS = cfg["crs"]

        logger = misc.format_logger(logger, LOG_FILE)
        logger.info("---------------------")
        logger.info(f"{LOG_FILE = }")
        logger.info(f"{PATCH_FOLDER = }")
        logger.info(f"{PATCH_CORR_FOLDER = }")
        logger.info(f"{MOSAIC_FOLDER = }")
        logger.info(f"{CRS = }")

        logger.info("Mosaicing border...")
        tic_mosaic = time.time()

        logger.info(F"Mosaicing border on {PATCH_FOLDER}...")
        tic_mosaic = time.time()
        mosaic_grid.mosaic_id(PATCH_FOLDER, os.path.join(MOSAIC_FOLDER,'opt_cut'), CRS)
        toc_mosaic = time.time()
        logger.info(f"...ended process after {toc_mosaic-tic_mosaic} s.\n")      
        
        if os.path.exists(PATCH_CORR_FOLDER):
            logger.info(F"Mosaicing border on {PATCH_CORR_FOLDER}...")
            tic_mosaic = time.time()
            mosaic_grid.mosaic_id(PATCH_CORR_FOLDER, os.path.join(MOSAIC_FOLDER,'corr'), CRS)
            toc_mosaic = time.time()
            logger.info(f"...ended process after {toc_mosaic-tic_mosaic} s.\n")           

    global_stop_time = time.time()
    logger.info(f"Ended pipeline, time elapsed: {round(global_stop_time - global_start_time)} seconds")
    logger.info(f" - inferring, time elapsed: {round(toc_infer - tic_infer)} seconds") if cst.INFER else None
    logger.info(f" - cutting border, time elapsed: {round(toc_cut - tic_cut)} seconds") if cst.CUT_BORDER else None
    logger.info(f" - correcting artefacts, time elapsed: {round(toc_arte - tic_arte)} seconds") if cst.CORRECT else None
    logger.info(f" - mosaicing, time elapsed: {round(toc_mosaic - tic_mosaic)} seconds") if cst.MOSAIC else None


