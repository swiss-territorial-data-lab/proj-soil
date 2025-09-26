import os
import sys
from loguru import logger
from tqdm import tqdm

import geopandas as gpd
import pandas as pd
import rasterio as rio
from glob import glob
from shapely.geometry import box, Polygon

from scripts.utilities import constants as cst
import scripts.utilities.from_bpo.fct_misc as misc
import scripts.utilities.from_bpo.fct_rasters as rasters

logger = misc.format_logger(logger)


def control_overlap(gdf1, gdf2, threshold=0.5, op='larger'):
    """Test the overlap between the geometries of two GeoDataFrames and return the ids of the 1st gdf passing the test.

    Args:
        gdf1 (GeoDataFrame): first GeoDataFrame
        gdf2 (GeoDataFrame): second GeoDataFrame
        threshold (float, optional): limit value. Defaults to 0.5.
        op (str, optional): operator to use in the test. Possible values are 'larger' and "lte" (larger than or equal to). Defaults to 'larger'.

    Returns:
        list: ids of the 1st gdf passing the test
    """
    
    gdf1['total_area'] = gdf1.area

    intersection_gdf = gpd.overlay(gdf1, gdf2, how="difference", keep_geom_type=True)
    intersection_gdf = intersection_gdf.dissolve('id', as_index=False)
    intersection_gdf['percentage_area_left'] = intersection_gdf.area / intersection_gdf.total_area
    if op=='larger':
        id_to_keep = intersection_gdf.loc[intersection_gdf.percentage_area_left > threshold, 'id'].unique().tolist()
    elif op=='lte':
        id_to_keep = intersection_gdf.loc[intersection_gdf.percentage_area_left <= threshold, 'id'].unique().tolist()
    else:
        logger.critical('Passed operator is unknow. Please pass "larger" or "lte" (= less than or equal to) as operator.')
        sys.exit(1)

    return id_to_keep


def main(tile_dir, overlap_info=None, tile_suffix='.tif', output_dir='outputs', subtiles=False):
    """Get the delimitation of the tiles in a directory

    Args:
        tile_dir (str): path to the directory containing the tiles
        overlap_info (str or DataFrame, optional): path to the DataFrame or DataFrame with the information about the overlap between tiles at each scale. 
            If None, overlap is 0. Defaults to None.
        tile_suffix (str, optional): suffix of the filename, which is the part coming after the tile number or id. Defaults to '.tif'.
        output_dir (str, optional): path to the output directory. Defaults to 'outputs'.
        subtiles (bool, optional): whether to generate the subtiles over each tile or not. Defaults to False.

    Returns:
        tiles_gdf: GeoDataFrame with the bounding box and the info of each tile
        nodata_gdf: GeoDataFrame with the nodata areas in the tile bounding boxes
        subtiles_gdf: GeoDataFrame with the bounding box and the info of each subtiles. If `subtiles`==None, returns None
        written_files: list of the written files
    """

    os.makedirs(output_dir, exist_ok=True)
    written_files = [] 

    output_path_tiles = os.path.join(output_dir, 'tiles.gpkg')
    output_path_nodata = os.path.join(output_dir, 'nodata_areas.gpkg')

    if not cst.OVERWRITE and os.path.exists(output_path_tiles) and os.path.exists(output_path_nodata):
        tiles_gdf = gpd.read_file(output_path_tiles)
        nodata_gdf=gpd.read_file(output_path_nodata)
        logger.info('Files for tiles already exist. Reading from disk...')

    else:
        logger.info('Read info for tiles...')
        tile_list = glob(os.path.join(tile_dir, '*'+tile_suffix))

        if len(tile_list) == 0:
            logger.critical('No tile in the tile directory.')
            sys.exit(1)

        logger.info('Create a geodataframe with tile info...')
        tiles_dict = {'id': [], 'name': [], 'number': [], 'scale': [], 'geometry': [],
                      'pixel_size_x': [], 'pixel_size_y': [], 'dimension': [], 'origin': [], 'max_dx': [], 'max_dy': []}
        nodata_gdf = gpd.GeoDataFrame()
        for tile in tqdm(tile_list, desc='Read tile info'):

            # Get name and id of the tile
            tile_name = os.path.basename(tile).rstrip(tile_suffix)
            tiles_dict['name'].append(tile_name)
            tiles_dict['id'].append(tile_name)
            tiles_dict['number'].append(tile_name)

            with rio.open(tile) as src:
                bounds = src.bounds
                # first_band = src.read(1)
                meta = src.meta

            # Set tile geometry
            geom = box(*bounds)
            tiles_dict['geometry'].append(geom)
            tiles_dict['origin'].append(str(rasters.get_bbox_origin(geom)))
            tile_size = (meta['width'], meta['height'])
            tiles_dict['dimension'].append(str(tile_size))

            # Guess tile scale
            perimeter = geom.length
            tile_scale = perimeter/4
            tiles_dict['scale'].append(tile_scale)
            
            # Set pixel size
            pixel_size_x = abs(meta['transform'][0])
            pixel_size_y = abs(meta['transform'][4])

            try:
                assert round(pixel_size_x, 5) == round(pixel_size_y, 5), f'The pixels are not square on tile {tile_name}: {round(pixel_size_x, 5)} x {round(pixel_size_y, 5)} m.'
            except AssertionError as e:
                print()
                logger.warning(e)

            tiles_dict['pixel_size_x'].append(pixel_size_x)
            tiles_dict['pixel_size_y'].append(pixel_size_y)

            # If no info on the plan scales, leave dx and dy to 0.
            if overlap_info:
                if isinstance(overlap_info, str):
                    overlap_info_df = pd.read_csv(overlap_info)
                elif isinstance(overlap_info, pd.DataFrame):
                    overlap_info_df = overlap_info
                else:
                    logger.error('Unrecognized format for the overlap info!')
                    sys.exit(1)
                max_dx = overlap_info_df.loc[overlap_info_df.scale==tile_scale, 'max_dx'].iloc[0]/pixel_size_x
                max_dy = overlap_info_df.loc[overlap_info_df.scale==tile_scale, 'max_dy'].iloc[0]/pixel_size_y
            else:
                max_dx = 0
                max_dy = 0
            tiles_dict['max_dx'].append(max_dx)
            tiles_dict['max_dy'].append(max_dy)

            # Transform nodata area into polygons
            # temp_gdf = rasters.no_data_to_polygons(first_band, meta['transform'], meta['nodata'])
            # temp_gdf = pad_geodataframe(temp_gdf, bounds, tile_size, max(pixel_size_x, pixel_size_y), cst.GRID_LARGE_TILES, cst.GRID_LARGE_TILES, max_dx, max_dy)
            # temp_gdf = temp_gdf.assign(tile_name=tile_name, scale=tile_scale)
            # nodata_gdf = pd.concat([nodata_gdf, temp_gdf], ignore_index=True)
            nodata_gdf=gpd.GeoDataFrame(geometry=[]).set_crs(crs='EPSG:2056')
            nodata_gdf['tile_name']=None

        tiles_gdf = gpd.GeoDataFrame(tiles_dict, crs='EPSG:2056')

        tiles_gdf.to_file(output_path_tiles)
        written_files.append(output_path_tiles)

        nodata_gdf.to_file(output_path_nodata)
        written_files.append(output_path_nodata)

    if subtiles:
       
        logger.info('Determine subtiles...')
        subtiles_gdf = gpd.GeoDataFrame()
        for tile in tqdm(tiles_gdf.itertuples(), desc='Define a grid to subdivide tiles', total=tiles_gdf.shape[0]):
            tile_infos = {
                'tile_size': tuple(map(int, tile.dimension.strip('()').split(', '))), 
                'tile_origin': tuple(map(float, tile.origin.strip('()').split(', '))), 
                'pixel_size_x': tile.pixel_size_x,
                'pixel_size_y': tile.pixel_size_y,
                'max_dx': cst.OVERLAP_LARGE_TILES*100,
                'max_dy': cst.OVERLAP_LARGE_TILES*100
            }
            nodata_subset_gdf = nodata_gdf[nodata_gdf.tile_name==tile.name].copy()

            # Make a large tiling grid to cover the image
            temp_gdf = rasters.grid_over_tile(grid_width=cst.GRID_LARGE_TILES, grid_height=cst.GRID_LARGE_TILES, **tile_infos)

            # Only keep tiles that do not overlap too much the nodata zone
            large_id_on_image = control_overlap(temp_gdf[['id', 'geometry']].copy(), nodata_subset_gdf, threshold=cst.OVERLAP_LARGE_TILES)
            large_subtiles_gdf = temp_gdf[temp_gdf.id.isin(large_id_on_image)].copy()
            large_subtiles_gdf.loc[:, 'id'] = [f'({subtile_id}, {str(tile.number)})' for subtile_id in large_subtiles_gdf.id] 
            large_subtiles_gdf['initial_tile'] = tile.name

            # if (tile.max_dx == 0) and (tile.max_dy == 0):
            #     # Make a smaller tiling grid to not lose too much data
            #     temp_gdf = rasters.grid_over_tile(grid_width=cst.GRID_SMALL_TILES, grid_height=cst.GRID_SMALL_TILES, **tile_infos)
            #     # Only keep smal subtiles not under a large one
            #     small_subtiles_gdf = gpd.overlay(temp_gdf, large_subtiles_gdf, how='difference', keep_geom_type=True)
            #     small_subtiles_gdf = small_subtiles_gdf[small_subtiles_gdf.area > (cst.GRID_LARGE_TILES*0.1)**2-1].copy() 
                
            #     if not small_subtiles_gdf.empty:
            #         # Only keep tiles that do not overlap too much the nodata zone
            #         small_id_on_image = control_overlap(small_subtiles_gdf[['id', 'geometry']].copy(), nodata_subset_gdf, threshold=cst.OVERLAP_SMALL_TILES)
            #         small_subtiles_gdf = small_subtiles_gdf[small_subtiles_gdf.id.isin(small_id_on_image)].copy()
            #         small_subtiles_gdf.loc[:, 'id'] = [f'({subtile_id}, {str(tile.number)})' for subtile_id in small_subtiles_gdf.id]
            #         small_subtiles_gdf['initial_tile'] = tile.name

            #         subtiles_gdf = pd.concat([subtiles_gdf, small_subtiles_gdf], ignore_index=True)
            
            subtiles_gdf = large_subtiles_gdf

        if cst.CLIP_OR_PAD_SUBTILES == 'clip':
            logger.info('The tiles are clipped to the image border.')
            subtiles_gdf = gpd.overlay(
                subtiles_gdf, tiles_gdf[['name', 'geometry']], 
                how="intersection", keep_geom_type=True
            )
            subtiles_gdf = subtiles_gdf.loc[subtiles_gdf.initial_tile == subtiles_gdf.name, ['id', 'initial_tile','centroid_x', 'centroid_y','col', 'row', 'minx', 'maxy', 'file_name','geometry']] 

        filepath = os.path.join(output_dir, 'subtiles.gpkg')
        subtiles_gdf.to_file(filepath)
        written_files.append(filepath)

    else:
        subtiles_gdf = None
    
    logger.success('Done determining the tiling!')
    return tiles_gdf, nodata_gdf, subtiles_gdf, written_files
    

def pad_geodataframe(gdf, tile_bounds, tile_size, pixel_size, grid_width=256, grid_height=256, max_dx=0, max_dy=0):
    """Extend the GeoDataFrame of the tile, definded by its bounding box, to match with a specified grid, 
    defined by its cell width, height, and overlapp, as well as the pixel size.
    Save the result in a GeoDataFrame.

    Args:
        gdf (GeoDataFrame): GeoDataFrame in which the result is saved
        tile_bounds (bounds): bounds of the tile
        tile_size (tuple): dimensions of the tile in pixels
        pixel_size (float): size of the pixel in cm
        grid_width (int, optional): number of pixels along the width of one grid cell. Defaults to 256.
        grid_height (int, optional): number of pixels along the hight of one grid cell. Defaults to 256.
        max_dx (int, optional): overlap in pixels along the width. Defaults to 0.
        max_dy (int, optional): overlap in pixels along the height. Defaults to 0.

    Returns:
        gdf: GeoDataFrame with two additional geometries corresponding to the padding on the top and on the right
    """

    min_x, min_y, max_x, max_y = tile_bounds
    tile_width, tile_height = tile_size
    number_cells_x, number_cells_y = rasters.get_grid_size(tile_size, grid_width, grid_height, max_dx, max_dy)

    # Get difference between grid size and tile size
    pad_width_px_x = number_cells_x * (grid_width - max_dx) + max_dx - tile_width
    pad_width_px_y = number_cells_y * (grid_height - max_dy) + max_dy - tile_height

    # Convert dimensions from pixels to meters
    pad_width_m_x = pad_width_px_x * pixel_size
    pad_width_m_y = pad_width_px_y * pixel_size

    # Pad on the top
    vertices = [(min_x, max_y),
                (max_x + pad_width_m_x, max_y),
                (max_x + pad_width_m_x, max_y + pad_width_m_y),
                (min_x, max_y + pad_width_m_y)]
    polygon_top = Polygon(vertices)
    
    # Pad on the right
    vertices = [(max_x, min_y),
                (max_x + pad_width_m_x, min_y),
                (max_x + pad_width_m_x, max_y ),
                (max_x, max_y)]
    polygon_right = Polygon(vertices)

    gdf = pd.concat([gdf, gpd.GeoDataFrame({'id_nodata_poly': [10001, 10002], 'geometry': [polygon_top, polygon_right]}, crs="EPSG:2056")], ignore_index=True)

    return gdf

# ------------------------------------------

if __name__ == "__main__":

    cfg = misc.get_config('prepare_data.py', "The script produce vector files with the delimitation of tiles and subtiles.")

    # Load input parameters
    WORKING_DIR = cfg['working_dir']
    OUTPUT_DIR = cfg['output_dir']['vectors']
    TILE_DIR = cfg['output_dir']['tiles']

    OVERLAP_INFO = cfg['overlap_info'] if 'overlap_info' in cfg.keys() else None

    os.chdir(WORKING_DIR)

    _, _, written_files = main(TILE_DIR,  OVERLAP_INFO, output_dir=OUTPUT_DIR, subtiles=True)

    print()
    logger.success("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.success(written_file)