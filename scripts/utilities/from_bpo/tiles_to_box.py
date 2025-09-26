import os
import sys
from loguru import logger
from tqdm import tqdm

import geopandas as gpd
import numpy as np
import rasterio
from glob import glob
from rasterio.mask import mask

import scripts.utilities.constants as cst
import scripts.utilities.from_bpo.fct_misc as misc
import scripts.utilities.from_bpo.fct_rasters as rasters

logger = misc.format_logger(logger)

def main(tile_dir, bboxes, output_dir='outputs', tile_suffix='.tif'):
    """Clip the tiles in a directory to the box geometries from a GeoDataFrame

    Args:
        tile_dir (str): path to the directory containing the tiles
        bboxes (str or GeoDataFrame): path to the GeoDataFrame or GeoDataFrame with the definition of the boxes
        output_dir (str, optional): path to the output directory. Defaults to 'outputs'.
        tile_suffix (str, optional): suffix of the filename, which is the part coming after the tile number or id. Defaults to '.tif'.
    """

    os.makedirs(output_dir, exist_ok=True)
    pad_tiles = False

    logger.info('Read bounding boxes...')
    if isinstance(bboxes, str):
        bboxes_gdf = gpd.read_file(bboxes)
        # Exclude areas added for symbol classification
        bboxes_gdf.loc[:,'Num_box'] = bboxes_gdf.Num_box.astype(int)
        bboxes_gdf = bboxes_gdf[bboxes_gdf.Num_box <= 35].copy()
        # Find tilepath matching initial plan number
        bboxes_gdf['tilepath'] = [
            tilepath 
            for num_plan in bboxes_gdf.Num_plan
            for tilepath in glob(os.path.join(tile_dir, '*' + tile_suffix)) 
            if tilepath.endswith(f'{num_plan[:6]}_{num_plan[6:]}' + tile_suffix)
        ]
    elif isinstance(bboxes, gpd.GeoDataFrame):
        bboxes_gdf = bboxes.copy()
        bboxes_gdf['tilepath'] = [os.path.join(tile_dir, f'{initial_tile}{tile_suffix}') for initial_tile in bboxes_gdf.initial_tile.to_numpy()]
        if cst.CLIP_OR_PAD_SUBTILES == 'pad':
            logger.info('Subtiles not entirely covered by the tile will be padded.')
            pad_tiles = True
    else:
        logger.critical(f'Only the paths and the GeoDataFrames are accepted for the bbox parameter. Passed type: {type(bboxes)}.')
        sys.exit(1)

    name_correspondence_list = []
    for bbox in tqdm(bboxes_gdf.itertuples(), desc='Clip tiles to the AOI of the bbox', total=bboxes_gdf.shape[0]):

        tilepath = bbox.tilepath
        new_name = f"{bbox.minx}_{bbox.maxy}_{bbox.row}_{bbox.col}.tif" 
        output_path = os.path.join(output_dir, new_name)

        if not cst.OVERWRITE and os.path.exists(output_path):
            continue

        if os.path.exists(tilepath):

            if not cst.OVERWRITE and os.path.exists(output_path):
                continue

            # Clip the tile
            with rasterio.open(tilepath) as src:
                out_image, out_transform, = mask(src, [bbox.geometry.buffer(-0.00001)], crop=True)
                out_meta = src.profile

            height = out_image.shape[1]
            width = out_image.shape[2]
            side_diff_w = cst.GRID_LARGE_TILES-width
            side_diff_h = cst.GRID_LARGE_TILES-height

            if pad_tiles and (side_diff_w > 0 or side_diff_h > 0):
                pad_side_w = ((0, 0), (0, 0), (0, side_diff_w)) if side_diff_w != cst.GRID_LARGE_TILES else ((0, 0), (0, 0), (0, 0))
                pad_side_h = ((0, 0), (side_diff_h, 0), (0, 0)) if side_diff_h != cst.GRID_LARGE_TILES else ((0, 0), (0, 0), (0, 0))
                pad_side = tuple((x1 + x2, y1 + y2) for (x1, y1), (x2, y2) in zip(pad_side_w, pad_side_h))

                out_image = np.pad(out_image, pad_width=pad_side, constant_values=0) # out_meta['nodata']
                height = cst.GRID_LARGE_TILES
                width = cst.GRID_LARGE_TILES
                out_transform = rasterio.Affine(0.1, 0.0, bbox.geometry.bounds[0], 
                                       0.0, -0.1, bbox.geometry.bounds[3])
            out_meta.update({"driver": "GTiff",
                 "height": height,
                 "width": width,
                 "transform": out_transform,
                 "photometric": 'YCbCr',
                 "compress": 'JPEG'})

            # Save the clipped tile in correspondence list and to file
            name_correspondence_list.append((os.path.basename(tilepath).rstrip(tile_suffix), new_name.rstrip('.tif')))

            if np.any(out_image): 
                with rasterio.open(output_path, "w", **out_meta) as dst:
                    dst.write(out_image)

        else:
            print()
            try:
                logger.warning(f"No tile correponding to plan {bbox.id}")
            except AttributeError:
                logger.warning(f"No tile correponding to plan {bbox.Num_plan}")

    if (len(name_correspondence_list) > 0) & output_dir.endswith('clipped_tiles'):
        misc.save_name_correspondence(name_correspondence_list, tile_dir, 'rgb_name', 'bbox_name')

    if len(name_correspondence_list) > 0:
        logger.success(f"Done clipping the tiles to the bboxes! The files were written in the folder {output_dir}. Let's check them out!")
    else:
        logger.success(f"Done clipping the tiles to the bboxes! All files were already present in folder.")
        

# ------------------------------------------

if __name__ == "__main__":

    cfg = misc.get_config('prepare_data.py', "The script clips the tiles to the given bboxes.")

    # Load input parameters
    WORKING_DIR = cfg['working_dir']
    OUTPUT_DIR_TILES = cfg['output_dir']['tiles']

    TILE_DIR = cfg['tile_dir']
    BBOX_PATH = cfg['bbox']

    os.chdir(WORKING_DIR)
    os.makedirs(OUTPUT_DIR_TILES, exist_ok=True)

    main(TILE_DIR, BBOX_PATH, OUTPUT_DIR_TILES)