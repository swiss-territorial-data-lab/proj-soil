import yaml
import os
import sys
from tqdm import tqdm

import argparse

from loguru import logger

import rasterio
from rasterio.merge import merge
from rasterio.features import rasterize, shapes
from shapely.geometry import shape

import geopandas as gpd 
import numpy as np 

import scripts.utilities.constants as cst
import scripts.utilities.correct_artefacts as correct_artefacts
import scripts.utilities.monitor_changes as monitor_changes

from multiprocessing import Pool
from functools import partial

from joblib import Parallel, delayed
import multiprocessing
# from tqdm_joblib import tqdm_joblib
import time

def append_src_if_exists(src_files_list, tag_list, raster_path, tag):
    if os.path.exists(raster_path):
        src = rasterio.open(raster_path)
        src_files_list.append(src)
        tag_list.append(tag)
    return src_files_list, tag_list

def check_vertical_neighbors(generic_raster_path, candidates, iter_range, col, r, src_files_to_mosaic, neighbor_list):
    for adj_px_r, neighbor in zip(iter_range,candidates):
        raster_path = generic_raster_path + f"{r + adj_px_r}_{col}.tif"
        src_files_to_mosaic, neighbor_list = append_src_if_exists(src_files_to_mosaic, neighbor_list, raster_path, neighbor)
    return src_files_to_mosaic, neighbor_list

def check_horizontal_neighbors(generic_raster_path, candidates, iter_range, row, c, src_files_to_mosaic, neighbor_list):
    for adj_px_c, neighbor in zip(iter_range,candidates):
        raster_path = generic_raster_path + f"{row}_{c + adj_px_c}.tif"
        src_files_to_mosaic, neighbor_list = append_src_if_exists(src_files_to_mosaic, neighbor_list, raster_path, neighbor)
    return src_files_to_mosaic, neighbor_list

def check_all_neighbors(generic_raster_path, candidates, iter_range_r, iter_range_c, r, c, src_files_to_mosaic, neighbor_list):
    for adj_px_r, neighbor_row in zip(iter_range_r, candidates):
        for adj_px_c, neighbor in zip(iter_range_c, neighbor_row):
            raster_path = generic_raster_path + f"{r + adj_px_r}_{c + adj_px_c}.tif"
            src_files_to_mosaic, neighbor_list = append_src_if_exists(src_files_to_mosaic, neighbor_list, raster_path, neighbor)
    return src_files_to_mosaic, neighbor_list        
    
def do_post_processing(file, INPUT_FOLDER_1, INPUT_FOLDER_2, CONF_FOLDER_1, CONF_FOLDER_2, CORRECTED_FOLDER_1, 
                       CORRECTED_FOLDER_2, CORRECTED_FOLDER_1_CB, CORRECTED_FOLDER_2_CB, MONITORED_FOLDER, CORRECTION_TYPE, TO_RECLASSIFY, RECLASSIFICATION_RULES, MAX_C, MAX_R, BLOCK_SIZE):
    
    if not file.endswith((".tif", ".tiff")) or cst.CORRECT and os.path.exists(os.path.join(CORRECTED_FOLDER_1,file)) and not cst.OVERWRITE:
        return 
    # file = '25759702_11969752_6_8.tif'
    coords = file.replace('.tif','').split('_')
    x, y, r, c = map(int, coords)
    src_files_to_mosaic = []
    src_files_to_mosaic_2 = []
    conf_files_to_mosaic_1 = []
    conf_files_to_mosaic_2 = []
    neighbor_list = []
    
    with rasterio.open(os.path.join(INPUT_FOLDER_1,file)) as src:

        if r==0:
            if c==0: #[top left corner]
                raster_path = os.path.join(INPUT_FOLDER_1,f"{x-BLOCK_SIZE}_{y+BLOCK_SIZE}_{MAX_R}_{MAX_C}.tif")
                src_files_to_mosaic, neighbor_list = append_src_if_exists(src_files_to_mosaic, neighbor_list, raster_path, 'n_nw')				
                
                generic_raster_path = os.path.join(INPUT_FOLDER_1, f"{x-BLOCK_SIZE}_{y}_")
                candidates = ['n_w', 'n_sw']
                src_files_to_mosaic, neighbor_list = check_vertical_neighbors(generic_raster_path, candidates, range(0, 2), MAX_C, 
                                                                              r, src_files_to_mosaic, neighbor_list)

                generic_raster_path = os.path.join(INPUT_FOLDER_1, f"{x}_{y+BLOCK_SIZE}_")
                candidates = ['n_n', 'n_ne']
                src_files_to_mosaic, neighbor_list = check_horizontal_neighbors(generic_raster_path, candidates, range(0, 2), MAX_R, 
                                                                                c, src_files_to_mosaic, neighbor_list)			
                                
                generic_raster_path = os.path.join(INPUT_FOLDER_1, f"{x}_{y}_")
                candidates = [['n','n_e'], ['n_s', 'n_se']]
                src_files_to_mosaic, neighbor_list = check_all_neighbors(generic_raster_path, candidates, range(0, 2), range(0, 2), 
                                                                         r, c, src_files_to_mosaic, neighbor_list)

            elif c==MAX_C: # [top right corner]
                raster_path = os.path.join(INPUT_FOLDER_1,f"{x+BLOCK_SIZE}_{y+BLOCK_SIZE}_{MAX_R}_{0}.tif")
                src_files_to_mosaic, neighbor_list = append_src_if_exists(src_files_to_mosaic, neighbor_list, raster_path, 'n_ne')
       
                generic_raster_path = os.path.join(INPUT_FOLDER_1, f"{x+BLOCK_SIZE}_{y}_")
                candidates =  ['n_e', 'n_se']
                src_files_to_mosaic, neighbor_list = check_vertical_neighbors(generic_raster_path, candidates, range(0, 2), 0, 
                                                                              r, src_files_to_mosaic, neighbor_list)
                        
                generic_raster_path = os.path.join(INPUT_FOLDER_1, f"{x}_{y+BLOCK_SIZE}_")
                candidates = ['n_nw', 'n_n']
                src_files_to_mosaic, neighbor_list = check_horizontal_neighbors(generic_raster_path, candidates, range(-1, 1), MAX_R, 
                                                                                c, src_files_to_mosaic, neighbor_list)	
                                
                generic_raster_path = os.path.join(INPUT_FOLDER_1, f"{x}_{y}_")
                candidates = [['n_w', 'n'], ['n_sw', 'n_s']]
                src_files_to_mosaic, neighbor_list = check_all_neighbors(generic_raster_path, candidates, range(0, 2), range(-1, 1), 
                                                                         r, c, src_files_to_mosaic, neighbor_list)
                             
            else: # [upper border]
                generic_raster_path = os.path.join(INPUT_FOLDER_1, f"{x}_{y+BLOCK_SIZE}_")
                candidates = ['n_nw', 'n_n', 'n_ne']
                src_files_to_mosaic, neighbor_list = check_horizontal_neighbors(generic_raster_path, candidates, range(-1, 2), MAX_R, 
                                                                                c, src_files_to_mosaic, neighbor_list)	

                generic_raster_path = os.path.join(INPUT_FOLDER_1, f"{x}_{y}_")
                candidates = [['n_w', 'n','n_e'], ['n_sw', 'n_s', 'n_se']]
                src_files_to_mosaic, neighbor_list = check_all_neighbors(generic_raster_path, candidates, range(0, 2), range(-1, 2), 
                                                                         r, c, src_files_to_mosaic, neighbor_list)
                
        elif r==MAX_R:
            if c==0: # [bottom left corner]
                raster_path = os.path.join(INPUT_FOLDER_1,f"{x-BLOCK_SIZE}_{y-BLOCK_SIZE}_{0}_{MAX_C}.tif")
                src_files_to_mosaic, neighbor_list = append_src_if_exists(src_files_to_mosaic, neighbor_list, raster_path, 'n_sw')

                generic_raster_path = os.path.join(INPUT_FOLDER_1, f"{x-BLOCK_SIZE}_{y}_")
                candidates = ['n_nw', 'n_w']
                src_files_to_mosaic, neighbor_list = check_vertical_neighbors(generic_raster_path, candidates, range(-1, 1), MAX_C, 
                                                                              r, src_files_to_mosaic, neighbor_list)
                        
                generic_raster_path = os.path.join(INPUT_FOLDER_1, f"{x}_{y-BLOCK_SIZE}_")
                candidates = ['n_s', 'n_se']
                src_files_to_mosaic, neighbor_list = check_horizontal_neighbors(generic_raster_path, candidates, range(0, 2), 0, 
                                                                                c, src_files_to_mosaic, neighbor_list)                		
                                
                generic_raster_path = os.path.join(INPUT_FOLDER_1, f"{x}_{y}_")
                candidates = [['n_n', 'n_ne'], ['n','n_e']]
                src_files_to_mosaic, neighbor_list = check_all_neighbors(generic_raster_path, candidates, range(-1, 1), range(0, 2), 
                                                                         r, c, src_files_to_mosaic, neighbor_list)

            elif c==MAX_C: # [bottom right corner]
                raster_path = os.path.join(INPUT_FOLDER_1,f"{x+BLOCK_SIZE}_{y-BLOCK_SIZE}_{0}_{0}.tif")
                src_files_to_mosaic, neighbor_list = append_src_if_exists(src_files_to_mosaic, neighbor_list, raster_path, 'n_se')
                
                generic_raster_path = os.path.join(INPUT_FOLDER_1, f"{x+BLOCK_SIZE}_{y}_")
                candidates =  ['n_ne', 'n_e']
                src_files_to_mosaic, neighbor_list = check_vertical_neighbors(generic_raster_path, candidates, range(-1, 1), 0, 
                                                                              r, src_files_to_mosaic, neighbor_list)
                        
                generic_raster_path = os.path.join(INPUT_FOLDER_1, f"{x}_{y-BLOCK_SIZE}_")
                candidates = ['n_sw', 'n_s']
                src_files_to_mosaic, neighbor_list = check_horizontal_neighbors(generic_raster_path, candidates, range(-1, 1), 0, 
                                                                                c, src_files_to_mosaic, neighbor_list)    
                                           
                generic_raster_path = os.path.join(INPUT_FOLDER_1, f"{x}_{y}_")
                candidates = [['n_nw', 'n_n'], ['n_w', 'n']]
                src_files_to_mosaic, neighbor_list = check_all_neighbors(generic_raster_path, candidates, range(-1, 1), range(-1, 1), 
                                                                         r, c, src_files_to_mosaic, neighbor_list)

            else: #[lower border]
                generic_raster_path = os.path.join(INPUT_FOLDER_1, f"{x}_{y-BLOCK_SIZE}_")
                candidates = ['n_sw', 'n_s', 'n_se']
                src_files_to_mosaic, neighbor_list = check_horizontal_neighbors(generic_raster_path, candidates, range(-1, 2), 0, 
                                                                                c, src_files_to_mosaic, neighbor_list)    

                generic_raster_path = os.path.join(INPUT_FOLDER_1, f"{x}_{y}_")
                candidates = [['n_nw', 'n_n', 'n_ne'], ['n_w', 'n','n_e']]
                src_files_to_mosaic, neighbor_list = check_all_neighbors(generic_raster_path, candidates, range(-1, 1), range(-1, 2), 
                                                                         r, c, src_files_to_mosaic, neighbor_list)                

        elif c==0: # tacit r!=0 and r!=R_MAX [left border] 
            generic_raster_path = os.path.join(INPUT_FOLDER_1, f"{x-BLOCK_SIZE}_{y}_")
            candidates = ['n_nw', 'n_w', 'n_sw']
            src_files_to_mosaic, neighbor_list = check_vertical_neighbors(generic_raster_path, candidates, range(-1, 2), MAX_C, 
                                                                            r, src_files_to_mosaic, neighbor_list)

            generic_raster_path = os.path.join(INPUT_FOLDER_1, f"{x}_{y}_")
            candidates = [['n_n', 'n_ne'], ['n','n_e'], ['n_s', 'n_se']]
            src_files_to_mosaic, neighbor_list = check_all_neighbors(generic_raster_path, candidates, range(-1, 2), range(0, 2), 
                                                                        r, c, src_files_to_mosaic, neighbor_list)          

        elif c==MAX_C:  # tacit r!=0 and r!=R_MAX [right border]
            generic_raster_path = os.path.join(INPUT_FOLDER_1, f"{x+BLOCK_SIZE}_{y}_")
            candidates = ['n_ne', 'n_e', 'n_se']
            src_files_to_mosaic, neighbor_list = check_vertical_neighbors(generic_raster_path, candidates, range(-1, 2), 0, 
                                                                            r, src_files_to_mosaic, neighbor_list)

            generic_raster_path = os.path.join(INPUT_FOLDER_1, f"{x}_{y}_")
            candidates = [['n_nw', 'n_n'], ['n_w', 'n'], ['n_sw', 'n_s']]
            src_files_to_mosaic, neighbor_list = check_all_neighbors(generic_raster_path, candidates, range(-1, 2), range(-1, 1), 
                                                                     r, c, src_files_to_mosaic, neighbor_list)        
            
        else:
            generic_raster_path = os.path.join(INPUT_FOLDER_1, f"{x}_{y}_")
            candidates = [['n_nw', 'n_n', 'n_ne'], ['n_w', 'n','n_e'], ['n_sw', 'n_s', 'n_se']]
            src_files_to_mosaic, neighbor_list = check_all_neighbors(generic_raster_path, candidates, range(-1, 2), range(-1, 2), 
                                                                     r, c, src_files_to_mosaic, neighbor_list)          

        buffer_w_min = src.width
        buffer_w_max = (src.width)*2
        buffer_h_min = src.height
        buffer_h_max = (src.height)*2
        if not any(elem in ['n_nw', 'n_w', 'n_sw'] for elem in neighbor_list):
            buffer_w_min = 0
            buffer_w_max = src.width
        if not any(elem in ['n_nw', 'n_n', 'n_ne'] for elem in neighbor_list):
            buffer_h_min = 0
            buffer_h_max = src.height

        out_profile = src.profile.copy()
        out_profile.update(count=1)

        mosaic, out_trans = merge(src_files_to_mosaic, indexes=[1] , output_count=1)
  
        if cst.CORRECT:
            mosaic_corr = correct_artefacts.main(mosaic, out_trans, TO_RECLASSIFY, RECLASSIFICATION_RULES)

            src_corr = mosaic_corr[buffer_h_min:buffer_h_max,buffer_w_min:buffer_w_max] 

            with rasterio.open(os.path.join(CORRECTED_FOLDER_1,file), "w", **out_profile) as dest:
                dest.write(src_corr,1)
                
            if CORRECTION_TYPE == 'combined':
                mosaic_corr_cb = correct_artefacts.main(mosaic_corr, out_trans, True, RECLASSIFICATION_RULES)

                src_corr_cb = mosaic_corr_cb[buffer_h_min:buffer_h_max,buffer_w_min:buffer_w_max] 

                with rasterio.open(os.path.join(CORRECTED_FOLDER_1_CB,file), "w", **out_profile) as dest:
                    dest.write(src_corr_cb,1)
                    
            
            if CORRECTED_FOLDER_2 is not None:
                if cst.OVERWRITE is True or len(os.listdir(CORRECTED_FOLDER_2))==0:
                    [append_src_if_exists(src_files_to_mosaic_2, neighbor_list, src_files_to_mosaic[el].name.replace(INPUT_FOLDER_1,INPUT_FOLDER_2), 
                                neighbor_list[el]) for el in range(len(src_files_to_mosaic))] 
                    mosaic_2, _ = merge(src_files_to_mosaic_2, indexes=[1] , output_count=1)
                    
                    mosaic_corr_2 = correct_artefacts.main(mosaic_2, out_trans, TO_RECLASSIFY, RECLASSIFICATION_RULES)
                    src_corr2 = mosaic_corr_2[buffer_h_min:buffer_h_max,buffer_w_min:buffer_w_max] 
                    with rasterio.open(os.path.join(CORRECTED_FOLDER_2,file), "w", **out_profile) as dest:
                        dest.write(src_corr2,1)

                    if CORRECTION_TYPE == 'combined':
                        mosaic_corr_2_cb = correct_artefacts.main(mosaic_corr_2, out_trans, True, RECLASSIFICATION_RULES)

                        src_corr_2_cb = mosaic_corr_2_cb[buffer_h_min:buffer_h_max,buffer_w_min:buffer_w_max] 

                        with rasterio.open(os.path.join(CORRECTED_FOLDER_2_CB,file), "w", **out_profile) as dest:
                            dest.write(src_corr_2_cb,1)
                else:
                    [append_src_if_exists(src_files_to_mosaic_2, neighbor_list, src_files_to_mosaic[el].name.replace(INPUT_FOLDER_1,CORRECTED_FOLDER_2),
                                          neighbor_list[el]) for el in range(len(src_files_to_mosaic))]                
                    mosaic_corr_2, _ = merge(src_files_to_mosaic_2, indexes=[1], output_count=1)                            
                            
        if MONITORED_FOLDER is not None : 

            [append_src_if_exists(conf_files_to_mosaic_1, neighbor_list, src_files_to_mosaic[el].name.replace(INPUT_FOLDER_1,CONF_FOLDER_1), 
                                  neighbor_list[el]) for el in range(len(src_files_to_mosaic))] 
            [append_src_if_exists(conf_files_to_mosaic_2, neighbor_list, src_files_to_mosaic[el].name.replace(INPUT_FOLDER_1,CONF_FOLDER_2), 
                                  neighbor_list[el]) for el in range(len(src_files_to_mosaic))]    
            
            if not conf_files_to_mosaic_2 :
                return

            mosaic_conf_1, _ = merge(conf_files_to_mosaic_1, indexes=[1] , output_count=1)
            mosaic_conf_2, _ = merge(conf_files_to_mosaic_2, indexes=[1] , output_count=1)   

            if cst.CORRECT:
                mosaic4moni_1 = mosaic_corr
                mosaic4moni_2 = mosaic_corr_2
            else:
                mosaic4moni_1 = mosaic
                [append_src_if_exists(src_files_to_mosaic_2, neighbor_list, src_files_to_mosaic[el].name.replace(INPUT_FOLDER_1,INPUT_FOLDER_2), 
                            neighbor_list[el]) for el in range(len(src_files_to_mosaic))] 
                
                mosaic4moni_2, _ = merge(src_files_to_mosaic_2, indexes=[1] , output_count=1)

            mosaic_diff = (mosaic4moni_1.astype(np.int16)-1)*17+mosaic4moni_2.astype(np.int16)
            mosaic_changes = monitor_changes.main(mosaic_diff, out_trans, mosaic_conf_1, mosaic_conf_2) 
            
            src_changes = mosaic_changes[buffer_h_min:buffer_h_max,buffer_w_min:buffer_w_max] 

            with rasterio.open(os.path.join(MONITORED_FOLDER,file), "w", **out_profile) as dest:
                dest.write(src_changes,1)

    return

def post_processing_embedding(INPUT_FOLDER_1, INPUT_FOLDER_2, CONF_FOLDER_1, CONF_FOLDER_2, CORRECTED_FOLDER_1, CORRECTED_FOLDER_2, 
                              MONITORED_FOLDER, CORRECTION_TYPE, RECLASSIFICATION_RULES):
    
    """
    Three cases of post-processing are handled:
        (1) At time t, the correction of artefacts: for use, see config-pp.yaml
            /!\ Constants to set:
                CORRECT = True
        (2) At time t+1, when there is already the corrected version of time t, correction of the artefacts at time t+1 and monitoring 
            between t and t+1: for use, see config/config-pp-monitoring.yaml    
            /!\ Constants to set:
                CORRECT = True
        (3) Monitoring between time t and t+1 (predictions corrected or not): for use see config/config-utilities.yaml
            /!\ Constants to set:
                CORRECT = False
    Parameters:
        INPUT_FOLDER_1 (str): path to
            (1) prediction of the model at time t
            (2) prediction of the model at time t
            (3) prediction or corrected prediction of the model at time t
        INPUT_FOLDER_2 (str): path of
            (1) 
            (2) prediction of the model at at time!=t
            (3) prediction or corrected prediction of the model at time!=t
        CONF_FOLDER_1 (str): path to
            (1) 
            (2) prediction probability of the model at time t
            (3) prediction probability of the model at time t
        CONF_FOLDER_2 (str): path to 
            (1) 
            (2) prediction probability of the model at time!=t
            (3) prediction probability of the model at time!=t
        CORRECTED_FOLDER_1 (str): path to
            (1) corrected prediction of the model at time t
            (2) corrected prediction of the model at time t
            (3) 
        CORRECTED_FOLDER_2 (str): path to
            (1) 
            (2) corrected prediction of the model at at time!=t
            (3) 
        MONITORED_FOLDER (str): path to
            (1) 
            (2) monitored changes
            (3) monitored changes 
        CORRECTION_TYPE (str): 
                  'mc': on the prediction classes, 
                  'soil': agreggate prediction into the superclasses before correcting
                  'combined': 1. 'mc', 2. 'soil' on 'mc' output. 
                   NB: corresponding subfolder are automatically created in the 'corrrected_folder_*'
        RECLASSIFICATION_RULES (dict): reclassification of raster values from 'mc' classes to 'soil' superclasses
    
    Returns:
        None
    """

    CORRECTED_FOLDER_1_CB = None
    CORRECTED_FOLDER_2_CB = None
    TO_RECLASSIFY = False
    if cst.CORRECT:
        if CORRECTION_TYPE not in ['combined', 'mc', 'soil']:
            logger.critical(f"{CORRECTION_TYPE} is not a recognized correction type. Check config file.")
            sys.exit(1)
        if CORRECTION_TYPE == 'combined':       
            CORRECTED_FOLDER_1_CB = os.path.join(CORRECTED_FOLDER_1,'combined')
            os.makedirs(CORRECTED_FOLDER_1_CB, exist_ok=True)
            if CORRECTED_FOLDER_2 is not None:
                CORRECTED_FOLDER_2_CB = os.path.join(CORRECTED_FOLDER_2,'combined')
                os.makedirs(CORRECTED_FOLDER_2_CB, exist_ok=True)    
        if CORRECTION_TYPE in ['combined', 'mc']:
            CORRECTED_FOLDER_1 = os.path.join(CORRECTED_FOLDER_1,'mc')
            if CORRECTED_FOLDER_2 is not None:
                CORRECTED_FOLDER_2 = os.path.join(CORRECTED_FOLDER_2,'mc')
        if CORRECTION_TYPE == 'soil':
            CORRECTED_FOLDER_1 = os.path.join(CORRECTED_FOLDER_1,'soil')
            if CORRECTED_FOLDER_2 is not None:
                CORRECTED_FOLDER_2 = os.path.join(CORRECTED_FOLDER_2,'soil')
            TO_RECLASSIFY = True

        os.makedirs(CORRECTED_FOLDER_1, exist_ok=True)
        if CORRECTED_FOLDER_2 is not None:
            os.makedirs(CORRECTED_FOLDER_2, exist_ok=True)    


    if MONITORED_FOLDER is not None:
        os.makedirs(MONITORED_FOLDER, exist_ok=True)
        if cst.CORRECT:
            os.makedirs(CORRECTED_FOLDER_2, exist_ok=True)
    
    list_files = os.listdir(INPUT_FOLDER_1)

    BLOCK_SIZE = int(cst.REGION_LENGTH*10)
    MAX_R = cst.MAX_R
    MAX_C = cst.MAX_C

    num_cores = multiprocessing.cpu_count()-1
    logger.info(f"Starting job on {num_cores} cores...") 
    partial_func = partial(do_post_processing,INPUT_FOLDER_1=INPUT_FOLDER_1, INPUT_FOLDER_2=INPUT_FOLDER_2, CONF_FOLDER_1=CONF_FOLDER_1, CONF_FOLDER_2=CONF_FOLDER_2, 
                           CORRECTED_FOLDER_1=CORRECTED_FOLDER_1, CORRECTED_FOLDER_2=CORRECTED_FOLDER_2,
                           CORRECTED_FOLDER_1_CB=CORRECTED_FOLDER_1_CB, CORRECTED_FOLDER_2_CB=CORRECTED_FOLDER_2_CB,
                           MONITORED_FOLDER=MONITORED_FOLDER, CORRECTION_TYPE=CORRECTION_TYPE, TO_RECLASSIFY=TO_RECLASSIFY, RECLASSIFICATION_RULES=RECLASSIFICATION_RULES, 
                           MAX_C=MAX_C, MAX_R=MAX_R, BLOCK_SIZE=BLOCK_SIZE)

    with Pool(num_cores, maxtasksperchild=100) as p:
        for _ in tqdm(p.imap(partial_func, list_files, chunksize=1), total=len(list_files)):
            pass

    # do_post_processing('25070000_11389752_19_11.tif',INPUT_FOLDER_1=INPUT_FOLDER_1, INPUT_FOLDER_2=INPUT_FOLDER_2, CONF_FOLDER_1=CONF_FOLDER_1, CONF_FOLDER_2=CONF_FOLDER_2, 
    #                        CORRECTED_FOLDER_1=CORRECTED_FOLDER_1, CORRECTED_FOLDER_2=CORRECTED_FOLDER_2,
    #                        CORRECTED_FOLDER_1_CB=CORRECTED_FOLDER_1_CB, CORRECTED_FOLDER_2_CB=CORRECTED_FOLDER_2_CB,
    #                        MONITORED_FOLDER=MONITORED_FOLDER, CORRECTION_TYPE=CORRECTION_TYPE, TO_RECLASSIFY=TO_RECLASSIFY, RECLASSIFICATION_RULES=RECLASSIFICATION_RULES, 
    #                        MAX_C=MAX_C, MAX_R=MAX_R, BLOCK_SIZE=BLOCK_SIZE)

    return


if __name__ == "__main__":
    t0 = time.time()

    # Argument and parameter specification
    parser = argparse.ArgumentParser(
        description="The script corrects the artefacts in the predictions.")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        default="/proj-soils/config/config-utilities.yaml")
    args = parser.parse_args()

    # load input parameters
    with open(args.config_file) as fp:
        cfg = yaml.load(
            fp,
            Loader=yaml.FullLoader)[os.path.basename(__file__)]
    
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

    # set up logger
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(LOG_FILE, level="INFO")

    logger.info(f"{CORRECTION_TYPE = }")
    logger.info(f"{RECLASSIFICATION_RULES = }")      
    logger.info(f"{INPUT_FOLDER_1 = }") 
    logger.info(f"{INPUT_FOLDER_2 = }") 
    logger.info(f"{CONF_FOLDER_1 = }")
    logger.info(f"{CONF_FOLDER_2 = }")
    logger.info(f"{CORRECTED_FOLDER_1 = }")
    logger.info(f"{CORRECTED_FOLDER_2 = }")
    logger.info(f"{MONITORED_FOLDER = }")

    logger.info("Started Programm")
    post_processing_embedding(INPUT_FOLDER_1, INPUT_FOLDER_2, CONF_FOLDER_1, CONF_FOLDER_2, CORRECTED_FOLDER_1, CORRECTED_FOLDER_2, 
                              MONITORED_FOLDER, CORRECTION_TYPE, RECLASSIFICATION_RULES)
    logger.info("Ended Program/n")

    logger.info(f"Post-processing done in {time.time() - t0:.2f}s.")
