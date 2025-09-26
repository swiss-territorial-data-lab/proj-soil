import os
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

from loguru import logger
import argparse
import yaml
import sys


def recursive_quadtree(rect, depth):
    """
    Recursively divides a rectangle into smaller rectangles based on the specified depth.

    Args:
        rect (tuple): The coordinates of the rectangle in the form (x0, y0, x1, y1).
        depth (int): The depth of the recursive division.

    Returns:
        list: A list of rectangles with their corresponding depth.
    """
    if depth == 0:
        return [[rect, depth]]

    x0, y0, x1, y1 = rect
    xmid = (x0 + x1) / 2
    ymid = (y0 + y1) / 2

    returnlist = [
        *recursive_quadtree((x0, y0, xmid, ymid), depth - 1),
        *recursive_quadtree((xmid, y0, x1, ymid), depth - 1),
        *recursive_quadtree((x0, ymid, xmid, y1), depth - 1),
        *recursive_quadtree((xmid, ymid, x1, y1), depth - 1),
        [rect, depth],
    ]

    return returnlist


def create_recursive_grid(SOURCE_FOLDER, TARGET_PATH, DEPTH, MAX_GRIDSIZE):
    """
    Creates a recursive grid from a set of polygons.

    Args:
        SOURCE_FOLDER (str): The path to the folder containing the input shapefiles.
        TARGET_PATH (str): The path to save the output grid shapefile.
        DEPTH (int): The depth of the recursive grid.
        MAX_GRIDSIZE (float): The maximum size of each grid cell.

    Returns:
        None
    """
    bboxes = []
    sub_aoi = []
    for file in os.listdir(SOURCE_FOLDER):
        if not file.endswith((".shp", ".gpkg")):
            continue
        gt = gpd.read_file(os.path.join(SOURCE_FOLDER, file))
        bboxes.append(list(gt.total_bounds))
        sub_aoi.append(file)

    # Create the grid
    grid = []
    unique_ids = []
    str_ids = []
    depths = []
    gridsizes = []
    sub_aoi_att = []

    unique_id = 0
    scale_ids = [0, 0, 0]

    for i, bbox in enumerate(bboxes):

        xshape = round((bbox[2] - bbox[0]) / MAX_GRIDSIZE)
        yshape = round((bbox[3] - bbox[1]) / MAX_GRIDSIZE)

        # bbox = xmin, ymin, xmax, ymax

        for y in np.arange(bbox[1], bbox[3], MAX_GRIDSIZE):
            for x in np.arange(bbox[0], bbox[2], MAX_GRIDSIZE):

                x, y = np.round(x, 1), np.round(y, 1)
                bounds = (x, y, x + MAX_GRIDSIZE, y + MAX_GRIDSIZE)

                # Ensure the last polygon does not exceed the bounding box
                if bounds[2] > bbox[2] + 1e-3 or bounds[3] > bbox[3] + 1e-3:
                    continue

                recursive_bounds = recursive_quadtree(bounds, depth=DEPTH)

                for coords, depth in recursive_bounds:

                    if not depth == 0:
                        scale_ids[:depth] = [0] * depth

                    poly = Polygon.from_bounds(*coords)

                    id_str = (
                        f"{'-'.join([str(el) for el in reversed(scale_ids[depth:])])}"
                    )
                    str_ids.append(id_str)

                    grid.append(poly)
                    unique_ids.append(unique_id)
                    depths.append(depth)
                    gridsizes.append(MAX_GRIDSIZE / 2**depth)
                    sub_aoi_att.append(sub_aoi[i])

                    scale_ids[depth] += 1
                    unique_id += 1

    # Convert grid to a GeoDataFrame
    grid_gdf = gpd.GeoDataFrame(
        {
            "geometry": grid,
            "unique_id": unique_ids,
            "str_ids": str_ids,
            "gridsize": gridsizes,
            "depth": depths,
            "sub_aoi": sub_aoi_att,
        },
        crs=2056,
    )

    assert np.unique(str_ids).shape[0] == len(grid_gdf)

    grid_gdf.to_file(TARGET_PATH)


if __name__ == "__main__":
    # Argument and parameter specification
    parser = argparse.ArgumentParser(
        description="The script creates a recursive grid from a set of polygons"
    )
    parser.add_argument(
        "-cfg",
        "--config_file",
        type=str,
        help="Framework configuration file",
        default="/proj-soils/config/config-utilities.yaml",
    )
    args = parser.parse_args()

    # load input parameters
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    SOURCE_FOLDER = cfg["source_folder"]
    TARGET_PATH = cfg["target_path"]
    DEPTH = cfg["depth"]
    MAX_GRIDSIZE = cfg["max_gridsize"]
    LOG_FILE = cfg["log_file"]

    # set up logger
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")
    logger.add(LOG_FILE, level="INFO")

    logger.info(f"{SOURCE_FOLDER = }")
    logger.info(f"{TARGET_PATH = }")
    logger.info(f"{DEPTH = }")
    logger.info(f"{MAX_GRIDSIZE = }")
    logger.info(f"{LOG_FILE = }")

    logger.info("Started Programm")
    create_recursive_grid(SOURCE_FOLDER, TARGET_PATH, DEPTH, MAX_GRIDSIZE)
    logger.info("Ended Program\n")
