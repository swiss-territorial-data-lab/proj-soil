import pandas as pd
import geopandas as gpd

from shapely.geometry import box

import rasterio

import os
import sys

import yaml
import argparse

from loguru import logger


def mosaic_contiguous(INPUT_FOLDER, OUTPUT_FOLDER, CRS, OTB_LOG, OTB_INSTALLATION, DTYPE, DEBUG=False):
    """
    Mosaic contiguous raster images.

    This function mosaics raster images that are contiguous in a specified folder
    using the Orfeo Toolbox (OTB). The resulting mosaicked images are saved to an
    output folder and compressed using the DEFLATE compression.

    Parameters:
    - INPUT_FOLDER (str): Path to the folder containing input raster images.
    - OUTPUT_FOLDER (str): Path to the output folder for mosaicked images.
    - CRS (int): Coordinate Reference System code.
    - OTB_LOG (str): Path to the OTB log-file.
    - OTB_INSTALLATION (str): Path to the OTB installation.
    - DTYPE (str): Data type of the output raster images.
    - DEBUG (bool, optional): Debug mode flag. If False, OTB output is silenced

    Returns:
    - None
    """
    idx = -1
    for file in os.listdir(INPUT_FOLDER):
        if not file.endswith(".tif"):
            continue
        idx += 1

        # create shapely box with tiff-bounds
        with rasterio.open(os.path.join(INPUT_FOLDER, file)) as src:
            bounds = src.bounds
            geom = box(*bounds)

        if idx == 0:
            # Initialize GeoDataFrame with the first image's information
            gdf = gpd.GeoDataFrame(
                {
                    "paths": os.path.join(INPUT_FOLDER, file), 
                    "geometry":[geom]
                },
                crs=2056)
            continue

        # Concatenate features for the subsequent images
        gdf = pd.concat((
            gdf,
            gpd.GeoDataFrame(
                {
                    # "id": idx,
                    "paths": os.path.join(INPUT_FOLDER, file),
                    "geometry" :[geom]
                },
                crs=2056)
        ))
    
    # dissolve contiguous features and aggregate the paths into a list
    gdf = (gdf
        .assign(geometry=gdf.geometry.buffer(0.0001))
        .dissolve()
        .explode(index_parts=False)
        .drop(["paths"], axis=1)
        .reset_index(drop=True)
        .sjoin(gdf)
        .reset_index()
        .dissolve(by="index", aggfunc=list)
        .drop(["index_right"], axis=1)
    )
    # pathlist: list of lists
    pathlist = list(gdf["paths"].values)


    mosaic_params = {
        "-harmo.method ": "none", # "band",
        "-nodata ": "0",
        "-ram ": "8192",
        "progress": "1"
        # "-comp.feather ": "large",
        # "-comp.feather.slim.length ": "50"
    }

    # one "region" is one set of contiguous tiffs
    for idx, region in enumerate(pathlist):

        assert isinstance(region, list)
        outputbase = os.path.join(OUTPUT_FOLDER, f"region{idx}")
        # skip existing files
        if os.path.exists(f"{outputbase}.tif"):
            continue       
        
        logger.info(f"Processing region {idx}")

        mosaic_params["-il "] = " ".join(region) + " "
        mosaic_params["-out "] = f"{outputbase}_first.tif {DTYPE}"

        # Construct OTB mosaic command
        no_warnings = f"export CPL_LOG={OTB_LOG}" # "set" -> Windows, "export" -> Mac
        mosaic_fn = os.path.join(OTB_INSTALLATION, "bin/otbcli_Mosaic")
        param_str = f"{' '.join([str(key)+str(value) for key, value in mosaic_params.items()])}"
        
        # print("before OTB")
        # Execute OTB mosaic command
        if DEBUG:
            os.system(f"{no_warnings}; {mosaic_fn} {param_str}")
        else:
            os.system(f"{no_warnings}; {mosaic_fn} {param_str} >> tmp/output.txt 2>&1") 

        # print("before translate")
        gdal_translate = os.path.join(OTB_INSTALLATION, "bin/gdal_translate")
        crs_params = f"-a_srs {CRS}"

        # Use GDAL translate to add the crs to the metadata of the mosaic
        # and remove intermediate file
        if DEBUG:
            os.system(f"{gdal_translate}  -co COMPRESS=DEFLATE {crs_params} {outputbase+'_first.tif'} {outputbase+'.tif'}")
        else:
            os.system(f"{gdal_translate}  -co COMPRESS=DEFLATE {crs_params} {outputbase+'_first.tif'} {outputbase+'.tif'} >> tmp/output.txt 2>&1") 
        os.remove(outputbase+'_first.tif')


if __name__ == "__main__":

    # Argument and parameter specification
    parser = argparse.ArgumentParser(
        description="The script vectorizes all tif-files from a \
            specified folder and adds some postprocessing steps")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        # default="proj-soils/config/config-eval.yaml")
        default="proj-soils/config/config-eval_heig-vd.yaml")
    args = parser.parse_args()

    # load input parameters
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    LOG_FILE = cfg["log_file"]
    INPUT_FOLDER = cfg["input_folder"]
    OUTPUT_FOLDER = cfg["output_folder"]
    CRS = cfg["crs"]
    OTB_LOG = cfg["otb_log"]
    OTB_INSTALLATION = cfg["otb_installation"]
    DTYPE = cfg["dtype"]

    # set up logger
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(LOG_FILE, level="INFO")

    logger.info(f"{LOG_FILE = }")
    logger.info(f"{INPUT_FOLDER = }")
    logger.info(f"{OUTPUT_FOLDER = }")
    logger.info(f"{CRS = }")
    logger.info(f"{OTB_LOG = }")
    logger.info(f"{OTB_INSTALLATION = }")
    logger.info(f"{DTYPE = }")


    logger.info("Started Programm")
    mosaic_contiguous(INPUT_FOLDER, OUTPUT_FOLDER, CRS, OTB_LOG, OTB_INSTALLATION, DTYPE)
    logger.info("Ended Program\n")
