import pandas as pd
import geopandas as gpd

import rasterio
from rasterio.mask import mask
from affine import Affine

import geopandas as gpd

import numpy as np
import numpy.ma as ma

import yaml
import os
import sys
import argparse
import time

from loguru import logger

import warnings
warnings.filterwarnings('ignore')


def cut_tiff_to_grid(TIFF_FOLDER, GRID_PATH, CELL_LENGTH, ID_COLUMN, OUT_FOLDER, FILE_EXT='.tif', MASK_PATH=None, GRID_QUERY=None) -> None:

    """
    Cut tiff images into grid cells based on a shapefile grid.
    
    This function processes tiff images based on a provided grid shapefile. It cuts
    the tiff images into grid cells, optionally using a mask shapefile to filter
    grid cells. The processed grid cells are saved as individual tiff images.

    Parameters:
    - TIFF_FOLDER (str): Path to the folder containing input tiff images.
    - GRID_PATH (str): Path to the grid shapefile.
    - CELL_LENGTH (int): Side length of the output grid cells.
    - ID_COLUMN (str): Column name in grid shapefile for grid cell identifiers.
    - OUT_FOLDER (str): Path to the output folder for processed grid cell tiff images.
    - MASK_PATH (str, optional): Path to the mask shapefile for filtering grid cells.
    - GRID_QUERY (str, optional): Query string for filtering grid cells.

    Returns:
    - None
    """
    # if a mask path is specified it is loaded. If it is a directory,
    # the included layers are concatenated
    if MASK_PATH is not None:
        if os.path.isdir(MASK_PATH):
            i = -1
            for file in os.listdir(MASK_PATH):
                if not (file.endswith(".gpkg") or file.endswith(".shp")):
                    continue
                i += 1
                if i == 0:
                    MASK = gpd.read_file(os.path.join(MASK_PATH, file))
                    continue
                MASK = pd.concat((
                    MASK,
                    gpd.read_file(os.path.join(MASK_PATH, file))
                ))
        else:
            MASK = gpd.read_file(MASK_PATH)

    for file in os.listdir(TIFF_FOLDER):

        if not file.endswith(FILE_EXT):
            continue

        with rasterio.open(os.path.join(TIFF_FOLDER, file)) as src:
            
            # for corner cases: shrink tiff boundaries by one pixel to
            # make sure no features that are merely touching are loaded
            conservative_bounds = np.array(src.bounds) + np.array([1, 1, -1, -1])
            

            grid = gpd.read_file(
                GRID_PATH,
                bbox=list(conservative_bounds))
            
            if GRID_QUERY is not None:
                grid = grid.query(GRID_QUERY)

            print(f'{len(grid) = }')
            if len(grid) == 0:
                continue


            logger.info(f"{file = }")
            logger.info(f"{len(grid)} gridcells on tiff-extent")
            grid.rename(columns={ID_COLUMN: "ID_COLUMN"}, inplace=True)


            if MASK_PATH is not None:
                # add new boolean column whether a gridcell intersects with the mask
                grid["intersects"] = grid["geometry"].intersects(MASK.dissolve()["geometry"].values[0])

            for gridcell in grid.itertuples():
                idx = gridcell[0]
                vector=grid.loc[[idx]] 

                if MASK_PATH is not None:
                    # skip if gridcell doesn't intersect with the mask
                    if not vector["intersects"].values[0]:
                        continue

                
                out_image, out_transform = mask(
                    src,
                    vector.geometry,
                    crop=True,
                    all_touched=True,
                    )
                
                # out_image = out_image[tuple(slice(np.min(idx), np.max(idx) + 1) for idx in np.where(out_image != 0))]


                # skip too small images (border images, that don't fill an entire gridcell)
                tolerance = 2
                if not ((abs(out_image.shape[1] - CELL_LENGTH) <= tolerance) and (abs(out_image.shape[2] - CELL_LENGTH) <= tolerance)):
                    print(f"Skipped too small image (out_image.shape = {out_image.shape})")
                    continue

                if not out_image.shape[1] == out_image.shape[2] == CELL_LENGTH:

                    image_xmin = out_transform[2]
                    image_ymax = out_transform[5]

                    
                    # if dimensions are too small, pad the array,
                    # if dimensions are too large, crop the array

                    # rows (y-axis)
                    diff = abs(out_image.shape[1] - CELL_LENGTH) # get the absolute difference between the row count and CELL_LENGTH
                    if out_image.shape[1] == CELL_LENGTH:
                        ypadding = (0,0)  # No padding needed if row count matches CELL_LENGTH
                    elif out_image.shape[1] > CELL_LENGTH:
                        ypadding = (0, 0)  # No padding needed if row count is greater than CELL_LENGTH
                        if image_ymax > vector.bounds.maxy.values[0]:
                            out_image = out_image[:, diff:, :]  # Crop from top if image extends beyond max y-bound
                        else:
                            out_image = out_image[:, :-diff, :]  # Crop from bottom otherwise
                    elif out_image.shape[1] < CELL_LENGTH:
                        if image_ymax < vector.bounds.maxy.values[0]:
                            ypadding = (diff, 0)  # Pad at top if image does not reach max y-bound
                        else:
                            ypadding = (0, diff)  # Pad at bottom otherwise

                    # columns (x-axis)
                    diff = abs(out_image.shape[2] - CELL_LENGTH) # get the absolute difference between the row count and CELL_LENGTH
                    if out_image.shape[2] == CELL_LENGTH:
                        xpadding = (0,0)  # No padding needed if column count matches CELL_LENGTH
                    elif out_image.shape[2] > CELL_LENGTH:
                        xpadding = (0, 0)  # No padding needed if column count is greater than CELL_LENGTH
                        if image_xmin < vector.bounds.minx.values[0]:
                            out_image = out_image[:, :, diff:]  # Crop from left if image extends beyond min x-bound
                        else:
                            out_image = out_image[:, :, :-diff]  # Crop from right otherwise
                    elif out_image.shape[2] < CELL_LENGTH:
                        if image_xmin > vector.bounds.minx.values[0]:
                            xpadding = (diff, 0)  # Pad at left if image does not reach min x-bound
                        else:
                            xpadding = (0, diff)  # Pad at right otherwise

                    # Define padding - (top, bottom), (left, right)
                    padding = ((0, 0), ypadding, xpadding)  # Adding 1 row at the bottom and 1 column on the right

                    # Pad the array
                    out_image = np.pad(out_image, padding, mode='constant', constant_values=0)

                # Update the affine transform to match the new bounds
                out_transform_list = list(out_transform)
                out_transform_list[2] = vector.bounds.minx.values[0]
                out_transform_list[5] = vector.bounds.maxy.values[0]
                out_transform = Affine(*out_transform_list)
                

                assert out_image.shape[1] == out_image.shape[2] == CELL_LENGTH
                assert np.round(out_transform[2], 1) == np.round(vector.bounds.minx.values[0], 1)
                assert np.round(out_transform[5], 1) == np.round(vector.bounds.maxy.values[0], 1)

                # Assert that there are no zeros in the image and 
                # interpolate if there are (max allowed = 1 pixel)
                if not out_image.all(): 
                    a = ma.masked_array(out_image, out_image==0)
                    for shift in (-1, 1):
                        for axis in (1, 2):        
                            a_shifted=np.roll(a, shift=shift, axis=axis)
                            idx = ~a_shifted.mask * a.mask
                            a[idx] = a_shifted[idx]

                    out_image = a * 1

                out_transform_list = list(out_transform)
                out_transform_list[2] = vector.bounds.minx.values[0]
                out_transform_list[5] = vector.bounds.maxy.values[0]
                out_transform = Affine(*out_transform_list)

                # update metadata 
                out_meta=src.meta.copy() # copy the metadata of the source DEM
                out_meta.update({
                    "driver": "Gtiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "nodata": 0
                })
                
                out_path = os.path.join(
                    OUT_FOLDER,
                    f"{str(gridcell.ID_COLUMN)}.tif"
                    )

                if os.path.exists(out_path):
                    os.remove(out_path)

                with rasterio.open(out_path, 'w', **out_meta) as dst:
                    dst.write(out_image)


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

    TIFF_FOLDER = cfg["tiff_folder"]
    GRID_PATH = cfg["grid_path"]
    GRID_QUERY = cfg["grid_query"]
    CELL_LENGTH = cfg["cell_length"]
    ID_COLUMN = cfg["id_column"]
    OUT_FOLDER = cfg["out_folder"]
    LOG_FILE = cfg["log_file"]
    MASK_PATH = cfg["mask_path"]
    FILE_EXT = cfg["im_file_ext"] 

    # set up logger
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")
    logger.add(LOG_FILE, level="INFO")

    logger.info(f"{TIFF_FOLDER = }")
    logger.info(f"{OUT_FOLDER = }")
    logger.info(f"{GRID_PATH = }")
    logger.info(f"{GRID_QUERY = }")
    logger.info(f"{CELL_LENGTH = }")
    logger.info(f"{ID_COLUMN = }")
    logger.info(f"{LOG_FILE = }")
    logger.info(f"{MASK_PATH = }")
    logger.info(f"{FILE_EXT = }")

    os.makedirs(OUT_FOLDER, exist_ok=True)

    tic =time.time()


    logger.info("Started Programm")
    cut_tiff_to_grid(TIFF_FOLDER, GRID_PATH, CELL_LENGTH, ID_COLUMN, OUT_FOLDER, FILE_EXT, MASK_PATH, GRID_QUERY)
    logger.info(f"Ended Program {time.time()-tic} s\n")