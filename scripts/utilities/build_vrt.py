
import os
import sys
import argparse
import yaml
import warnings
import time
import rasterio
import geopandas as gpd
warnings.filterwarnings("ignore")

from osgeo import gdal

from loguru import logger
from shapely import box


def vrt_build(SOURCE_FOLDER, TARGET_FOLDER, VRT_NAME, bbox_gpkg=None):
    """
    Build the VRT with the TIF files in the input folder.
    Optionnaly, if bbox_gpgk is defined, saved a vector file of the bounding box. 
    
    Parameters:
        SOURCE_FOLDER (str): Path to the source folder containing raster files.
        TARGET_FOLDER (str): Path to the target folder to save the VRT.
        VRT_NAME (str): Name of the output VRT file.
        bbox_gpkg (str): Name of the output BBOX file (in GPKG format).
    
    Returns:
        None
    """

    logger.info(f"Listing the image files...")

    list_im = [os.path.join(SOURCE_FOLDER,file) for file in os.listdir(SOURCE_FOLDER) if file.endswith(("tif", "tiff"))]

    vrt_options = gdal.BuildVRTOptions(resampleAlg='cubic', addAlpha=False, outputSRS='EPSG:2056')

    os.makedirs(TARGET_FOLDER, exist_ok=True)

    vrt_filename = os.path.join(TARGET_FOLDER, VRT_NAME)
    my_vrt = gdal.BuildVRT(vrt_filename, list_im, options=vrt_options)
    my_vrt = None
    
    if bbox_gpkg:
        with rasterio.open(vrt_filename) as src:
            bbox = box(*src.bounds)
            bbox_gdf = gpd.GeoDataFrame(data={'vrt': [vrt_filename]}, geometry=[bbox], crs=src.crs)
            bbox_gdf.to_file(os.path.join(TARGET_FOLDER, bbox_gpkg), driver="GPKG")

if __name__ == "__main__":
    # Argument and parameter specification
    parser = argparse.ArgumentParser(
        description="The script build a VRT.")
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

    SOURCE_FOLDER = cfg["source_folder"]
    TARGET_FOLDER = cfg["target_folder"]
    VRT_NAME = cfg["vrt_name"]
    BBOX_GPKG = cfg.get("bbox_gpkg", None)
    LOG_FILE = cfg["log_file"] 

    # set up logger
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(LOG_FILE, level="INFO")
    
    logger.info("---------------------")
    logger.info(f"{SOURCE_FOLDER = }")
    logger.info(f"{TARGET_FOLDER = }")
    logger.info(f"{VRT_NAME = }")
    logger.info(f"{BBOX_GPKG = }")

    tic=time.time()

    logger.info("Started Programm")
    vrt_build(SOURCE_FOLDER, TARGET_FOLDER, VRT_NAME, BBOX_GPKG)
    logger.info(f"Ended Program  {time.time()-tic} s.\n")