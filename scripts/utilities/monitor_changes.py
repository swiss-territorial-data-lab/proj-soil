import yaml
import os
import sys
import argparse
import time
from loguru import logger

import rasterio
from rasterstats import zonal_stats
import pandas as pd
import numpy as np

import fct_misc as misc
import constants as cst


def main(raster_diff, out_trans, raster_conf_1, raster_conf_2):
    """
    Vecotrize a raster of the difference in prediction between two years. Enrich the obtained
    vector with geometric descriptors and minimum prediciton probability index. Then, condition 
    are applied on each polygon (i.e. each transition) to sort them in different group. 

    Parameters:
        raster_diff (str): raster of the difference in prediction between two years. 
                           Typically: values between 0 and 255 for 16 classes.
        out_trans (str): transform information of the raster_diff 
        raster_conf_1 (str): raster of prediction probability index of the reference year
        raster_conf_2 (str): raster of prediciton probability index of the year to compare 
    
    Returns:
        None
    """

    pred_column = "DN"
    change_gdf = misc.raster2gdf(raster_diff, out_trans, pred_column, mask=None)
    
    change_gdf['change'] = 0 # no diff or negligible diff
    change_gdf['area'] = change_gdf.area
    change_gdf['bbox_fill_ratio']=change_gdf.area/change_gdf.geometry.envelope.area
    change_gdf['mrr'] = change_gdf.geometry.apply(lambda geom: geom.minimum_rotated_rectangle)
    change_gdf['coords'] = list(change_gdf['mrr'].apply(lambda geom: geom.exterior.coords))
    change_gdf['edges'] = change_gdf['coords'].apply(lambda row: [np.linalg.norm(np.array(row[i]) - np.array(row[i + 1])) for i in range(4)])
    change_gdf['long_axis'] = change_gdf['edges'].apply(lambda row: max(row))
    change_gdf['short_axis'] = change_gdf['edges'].apply(lambda row: min(row))
    change_gdf['axis_ratio'] = change_gdf.short_axis/change_gdf.long_axis
    change_gdf['rectangularity'] = change_gdf.area/(change_gdf.short_axis*change_gdf.long_axis)
    
    change_gdf['conf_1']= pd.DataFrame(zonal_stats(change_gdf, raster_conf_1[0], affine=out_trans, nodata=-999, stats=['mean']))
    change_gdf['conf_2']= pd.DataFrame(zonal_stats(change_gdf, raster_conf_2[0], affine=out_trans, nodata=-999, stats=['mean']))
    change_gdf['conf_min'] = change_gdf[['conf_1','conf_2']].min(axis=1)
    change_gdf.drop(columns=['mrr','coords', 'edges'], inplace=True)

    for row in change_gdf.itertuples():
        idx = row[0]
        COND_BUILDING = row.rectangularity > cst.RECTNESS_BUILDING and row.area> cst.AREA_BUILDING
        COND_TILT_BLDG = (row.axis_ratio<cst.LENGTHNESS_TILT or row.rectangularity<cst.RECTNESS_TILT)
        COND_TILT_HOVER = (row.axis_ratio<cst.LENGTHNESS_TILT or row.rectangularity<cst.RECTNESS_TILT) and row.short_axis<4
        COND_ARTEFACTS = row.bbox_fill_ratio > 0.8 and row.area> 2500
        COND_BORDER = row.axis_ratio<0.1 and row.short_axis<2
        COND_LOW = row.conf_min < 0.075

        if row.DN not in cst.SAME:
            if row.area < 1 :
                change_gdf.at[idx, 'change'] = 60 # noise
            else:
                if row.DN in cst.SNOW:
                    change_gdf.at[idx, 'change'] = 50 # diff due to snow cover
                elif row.DN in cst.SOIL_TO_NONSOIL:
                    change_gdf.at[idx, 'change'] = 20
                    if COND_LOW:
                        change_gdf.at[idx, 'change'] = 21 # low conf
                    elif row.DN in cst.SOIL_TO_BUILDING:
                        if COND_TILT_BLDG:
                            change_gdf.at[idx, 'change'] = 23 # Tilt building, no real changes because of rectangularity
                        elif COND_BUILDING:
                            change_gdf.at[idx, 'change'] = 22 # New constructions because of size and rectangularity  
                        else:
                            change_gdf.at[idx, 'change'] = 24
                    elif COND_BORDER:
                        change_gdf.at[idx, 'change'] = 40 # Border                   
                    elif row.DN in cst.VEGE_TO_NONSOIL and COND_TILT_HOVER:
                        change_gdf.at[idx, 'change'] = 25 # Hovering/Tilt vegetation, no real changes because of axis ratio
                    elif row.DN in cst.VEGE_WATER_ARTEFACTS and COND_ARTEFACTS:
                        change_gdf.at[idx, 'change'] = 26    

                elif row.DN in cst.NONSOIL_TO_SOIL:
                    change_gdf.at[idx, 'change'] = 30 
                    if COND_LOW:
                        change_gdf.at[idx, 'change'] = 31 # low conf
                    elif row.DN in cst.BUILDING_TO_SOIL:
                        if COND_TILT_BLDG:
                            change_gdf.at[idx, 'change'] = 33 # Tilt building, no real changes because of rectangularity
                        elif COND_BUILDING:
                            change_gdf.at[idx, 'change'] = 32 # Destructions because of size and rectangularity
                        else: 
                            change_gdf.at[idx, 'change'] = 34
                    elif COND_BORDER:
                        change_gdf.at[idx, 'change'] = 40 # Border
                    elif row.DN in cst.NONSOIL_TO_VEGE and COND_TILT_HOVER:
                        change_gdf.at[idx, 'change'] = 35 # Hovering/Tilt vegetation, no real changes because of axis ratio
                    elif row.DN in cst.WATER_VEGE_ARTEFACTS and COND_ARTEFACTS:
                        change_gdf.at[idx, 'change'] = 36
                else:
                    change_gdf.at[idx, 'change']= 10 # non-important


    out_shape = (round((change_gdf.total_bounds[3]-change_gdf.total_bounds[1])/0.1), round((change_gdf.total_bounds[2]-change_gdf.total_bounds[0])/0.1))
    transform = rasterio.transform.from_bounds(*change_gdf.total_bounds, out_shape[1], out_shape[0])

    raster = misc.gdf2raster(change_gdf, change_gdf.columns.get_loc('change')+1, out_shape, transform)

    return raster


if __name__ == "__main__":

    # Argument and parameter specification
    parser = argparse.ArgumentParser(
        description="The script corrects the artefacts in the predictions per individual tile.")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        default="/proj-soils/config/config-utilities.yaml")
    args = parser.parse_args()

    # load input parameters
    with open(args.config_file) as fp:
        cfg = yaml.load(
            fp,
            Loader=yaml.FullLoader)[os.path.basename(__file__)]

    CORR_PRED_COLUMN = cfg["corr_pred_column"]
    TO_RECLASSIFY = cfg["to_reclassify"]
    RECLASSIFICATION_RULES = cfg["reclassification_rules"]
    INPUT_FOLDER = cfg["input_folder"]
    OUTPUT_FOLDER = cfg["output_folder"]
    LOG_FILE = cfg["log_file"]

    # set up logger
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(LOG_FILE, level="INFO")

    logger.info(f"{CORR_PRED_COLUMN = }")     
    logger.info(f"{TO_RECLASSIFY = }")
    logger.info(f"{RECLASSIFICATION_RULES = }")      
    logger.info(f"{INPUT_FOLDER = }") 
    logger.info(f"{OUTPUT_FOLDER = }")
    logger.info

    tic = time.time()
    logger.info("Started Programm")

    for file in os.listdir(INPUT_FOLDER):
        if not file.endswith(("tif", "tiff")):
            continue

        raster_path = os.path.join(INPUT_FOLDER, file)
        with rasterio.open(raster_path) as src:
            src_data = src.read(1)
            transform = src.transform
            src_corr = main(src_data, transform, TO_RECLASSIFY, RECLASSIFICATION_RULES,file)

            out_profile = src.profile.copy()
            with rasterio.open(os.path.join(OUTPUT_FOLDER,file), "w", **out_profile) as dest:
                dest.write(src_corr,1)
    
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")
