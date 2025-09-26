import os
import sys
from argparse import ArgumentParser
from loguru import logger
from time import time
from yaml import load, FullLoader

import geopandas as gpd

import get_delimitation_tiles, tiles_to_box, fct_misc

# Processing ---------------------------------------

# Start chronometer
tic = time()
logger.info('Starting...')

cfg = fct_misc.get_config(os.path.basename(__file__), desc="The script prepares the initial files for the use of the OD in the detection of border points.")

# Load input parameters
WORKING_DIR = cfg['working_dir']
OUTPUT_DIR_VECT= cfg['output_dir']['vectors']
CREATE_GRID = cfg['create_grid']

TILE_DIR = cfg['tile_dir']

OVERLAP_INFO = cfg['overlap_info'] if 'overlap_info' in cfg.keys() else None
TILE_SUFFIX = cfg['tile_suffix'] if 'tile_suffix' in cfg.keys() else '.tif'

LOG_FILE = cfg['log_file'] 
logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add(LOG_FILE, level="INFO")

os.chdir(WORKING_DIR)

if CREATE_GRID:
    tiles_gdf, _, subtiles_gdf, written_files = get_delimitation_tiles.main(TILE_DIR, 
                                                                            overlap_info=OVERLAP_INFO, 
                                                                            tile_suffix=TILE_SUFFIX, 
                                                                            output_dir=OUTPUT_DIR_VECT, 
                                                                            subtiles=True)

    subtiles_gdf.to_file(os.path.join(OUTPUT_DIR_VECT, 'subtiles.gpkg'))
else:
    written_files = []
    subtiles_gdf = gpd.read_file(os.path.join(OUTPUT_DIR_VECT, 'subtiles.gpkg'))

# Clip images to subtiles
SUBTILE_DIR = os.path.join(TILE_DIR, 'subtiles')
os.makedirs(SUBTILE_DIR, exist_ok=True)
tiles_to_box.main(TILE_DIR, subtiles_gdf, SUBTILE_DIR, tile_suffix=TILE_SUFFIX,)

logger.success("The following files were written. Let's check them out!")
for written_file in written_files:
    logger.success(written_file)

# Stop chronometer
toc = time()
logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

sys.stderr.flush()