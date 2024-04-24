
import geopandas as gpd

import rasterio
from rasterio import features
from rasterio import Affine, CRS

import numpy as np

import yaml
import os
import sys

import argparse

from loguru import logger


def rasterize(POLYGON_FOLDER, FIELD, OUT_TIFF_FOLDER, CLASS_MAPPING, MASK_PATH=None) -> None:
    """
    Rasterizes polygon data onto a GeoTIFF.

    Args:
        POLYGON_FOLDER (str): Path to the input polygon shapefile.
        FIELD (str): Field name in the polygon attribute table containing class information.
        OUT_TIFF_FOLDER (str): Path to save the output rasterized GeoTIFF files.
        CLASS_MAPPING (dict): A dictionary mapping class labels from the polygon field to class numbers.
        MASK_PATH (str): Path to the mask shapefile used for clipping the input polygons.

    Returns:
        None
    """
    for root, dir, files in os.walk(POLYGON_FOLDER):

        for file in files:
            if not (file.endswith(".shp") or file.endswith(".gpkg")):
                continue
            
            # Clip polygons using the mask shapefile if provided, or read the shapefile directly
            if MASK_PATH is not None:
                gt = (gpd
                    .read_file(os.path.join(root, file))
                    .clip(gpd.read_file(MASK_PATH))
                )
            else:
                gt = gpd.read_file(os.path.join(root, file))

            if CLASS_MAPPING is None:
                gt[FIELD] = gt[FIELD].astype(int)
            else:
                # Map class labels to class numbers based on the provided mapping
                gt[FIELD] = [CLASS_MAPPING[clas] for clas in gt[FIELD]]

            # if the class is not a string, it is a nan and thus a 255 (nodata)
            # gt[FIELD] = [CLASS_MAPPING[clas] if isinstance(clas, str) else 0 for clas in gt[FIELD]]

            # Calculate raster dimensions based on polygon bounds and resolution
            xmin, ymin, xmax, ymax = gt.total_bounds
            width = round((xmax - xmin) / 0.1)
            height = round((ymax - ymin) / 0.1)


            meta = {
                'driver': 'GTiff',
                'dtype': 'int8',
                'nodata': 0,
                'width': width,
                'height': height,
                'count': 1,
                'crs': CRS.from_epsg(2056),
                'transform': Affine(
                    0.1, 0.0, np.round(xmin, 1),
                    0.0, -0.1, np.round(ymax, 1))
            }
            # keep initial folderstructure by creating same subfolders in new directory
            out_subfolder = root.replace(POLYGON_FOLDER, "").strip("/")
            out_folder = os.path.join(OUT_TIFF_FOLDER, out_subfolder)
            if not os.path.exists(out_folder):
                os.mkdir(out_folder)
            out_file = os.path.join(out_folder, file.rstrip(".shp")).rstrip(".gpkg") + ".tif"

            with rasterio.open(out_file, 'w', **meta) as dst:
                shapes = ((geom,value) for geom, value in zip(gt.geometry, gt[FIELD]))
                burned = features.rasterize(
                    shapes=shapes,
                    out=np.zeros((height, width)),
                    transform=dst.transform)


                dst.write_band(1, burned)
                

if __name__ == "__main__":
    # Argument and parameter specification
    parser = argparse.ArgumentParser(
        description="The script rasterizes a given polygon")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        default="proj-soils/config/config-eval_gt.yaml")
    args = parser.parse_args()

    # load input parameters
    with open(args.config_file) as fp:
        cfg = yaml.load(
            fp,
            Loader=yaml.FullLoader)[os.path.basename(__file__)]

    POLYGON_FOLDER = cfg["polygon_folder"]
    MASK_PATH = cfg["mask_path"]
    OUT_TIFF_FOLDER = cfg["out_tiff_folder"]
    FIELD = cfg["field"]
    CLASS_MAPPING = cfg["class_mapping"]
    LOG_FILE = cfg["log_file"]


    if not os.path.exists(OUT_TIFF_FOLDER):
        os.makedirs(OUT_TIFF_FOLDER)
        print(f"The directory {OUT_TIFF_FOLDER} was created.")

    # set up logger
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")
    logger.add(LOG_FILE, level="INFO")

    logger.info(f"{LOG_FILE = }")
    logger.info(f"{POLYGON_FOLDER = }")
    logger.info(f"{FIELD = }")
    logger.info(f"{OUT_TIFF_FOLDER = }")
    logger.info(f"{CLASS_MAPPING = }")
    logger.info(f"{MASK_PATH = }")

    logger.info("Started Programm")
    rasterize(POLYGON_FOLDER, FIELD, OUT_TIFF_FOLDER, CLASS_MAPPING, MASK_PATH)
    logger.info("Ended Program\n")