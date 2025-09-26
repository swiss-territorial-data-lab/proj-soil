import os
import sys

import yaml
import argparse

import rasterio
import geopandas as gpd

from rasterio.merge import merge
from tqdm import tqdm
from loguru import logger
from glob import glob


def mosaic_id(INPUT_FOLDER, OUTPUT_FOLDER, CRS, bbox_gpkg=None):
    """
    Mosaic contiguous raster for a 1km by 1km tiling scheme

    Parameters:
    - INPUT_FOLDER (str): Path to the folder containing input raster images.
    - OUTPUT_FOLDER (str): Path to the output folder for mosaicked images.
    - CRS (int): Coordinate Reference System code.
    - bbox_gpkg (str, optional): original BBOX of the input dataset.

    Returns:
    - None
    """

    if bbox_gpkg:
        bbox_gdf = gpd.read_file(bbox_gpkg)
        minx, miny, maxx, maxy = bbox_gdf.total_bounds

    list_ids = []
    tile_dict = {}
    list_files = [os.path.relpath(x, start=INPUT_FOLDER) for x in glob(f"{INPUT_FOLDER}/*.tif", recursive=True) + 
                  glob(f"{INPUT_FOLDER}/**/*.tif", recursive=True) + glob(f"{INPUT_FOLDER}/**/*.tiff", recursive=True)]

    for f in list_files:

        filename = os.path.basename(f)
        dirname = os.path.dirname(f)
        x = filename[0:5]
        y = filename[9:14]  
        tile_id = x + '_' + y

        k = f"{dirname}/{tile_id}"
        if k not in tile_dict.keys():
            tile_dict[k] = {
                "id": tile_id,
                "files": []
            }

        tile_dict[k]['files'].append(f)

    for k, tile in tqdm(tile_dict.items()):

        src_files_to_mosaic = []
        out_dir = os.path.dirname(k) if os.path.dirname(k) != '/' else ''
        outputbase = os.path.join(OUTPUT_FOLDER, out_dir, f"region_{tile['id']}.tif")
        os.makedirs(os.path.dirname(outputbase), exist_ok=True)

        region_minx = None

        for f in tile['files']:
            src = rasterio.open(os.path.join(INPUT_FOLDER, f))
            src_files_to_mosaic.append(src)
            if bbox_gpkg:
                _minx, _miny, _maxx, _maxy = src.bounds
                if not region_minx:
                    # 1st time => init
                    region_minx = _minx
                    region_miny = _miny
                    region_maxx = _maxx
                    region_maxy = _maxy
                else:
                    region_minx = min(region_minx, _minx)
                    region_miny = min(region_miny, _miny)
                    region_maxx = max(region_maxx, _maxx)
                    region_maxy = max(region_maxy, _maxy)


        if bbox_gpkg:
            output_minx = max(minx, region_minx)
            output_maxx = min(maxx, region_maxx)
            output_miny = max(miny, region_miny)
            output_maxy = min(maxy, region_maxy)
            bounds = [output_minx, output_miny, output_maxx, output_maxy]
        else:
            bounds = None

        mosaic, out_trans = merge(src_files_to_mosaic, bounds=bounds)

        out_profile = src.profile.copy()
        out_profile.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans
        })

        with rasterio.open(outputbase, "w", **out_profile) as dest:
            dest.write(mosaic)


if __name__ == "__main__":
    # Argument and parameter specification
    parser = argparse.ArgumentParser(
        description="The script vectorizes all tif-files from a \
            specified folder and adds some postprocessing steps")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        # default="proj-soils/config/config-eval.yaml")
        default="/proj-soils/config/config-pp.yaml")
    args = parser.parse_args()

    # load input parameters
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    LOG_FILE = cfg["log_file"]
    INPUT_FOLDER = cfg["input_folder"]
    OUTPUT_FOLDER = cfg["output_folder"]
    BBOX_GPKG = cfg.get("bbox_gpkg", None)
    CRS = cfg["crs"]

    # set up logger
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(LOG_FILE, level="INFO")

    logger.info(f"{LOG_FILE = }")
    logger.info(f"{INPUT_FOLDER = }")
    logger.info(f"{OUTPUT_FOLDER = }")
    logger.info(f"{CRS = }")

    logger.info("Started Programm")
    mosaic_id(INPUT_FOLDER, OUTPUT_FOLDER, CRS, BBOX_GPKG)
    logger.info("Ended Program\n")