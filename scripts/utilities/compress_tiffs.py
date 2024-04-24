import rasterio
import os
import argparse
import yaml
import sys
from loguru import logger

def compress_tiffs(SOURCE_FOLDER, TARGET_FOLDER, COMPRESSION_METHOD):
    """
    Compresses tiff files in the specified source folder and saves them
    in the target folder with the specified compression.

    Parameters:
        SOURCE_FOLDER (str): Path to the source folder containing raster files.
        TARGET_FOLDER (str): Path to the target folder to save the compressed raster files.
        COMPRESSION_METHOD (str): Compression method to be used.

    Returns:
        None
    """

    allowed_methods = ["ccittfax3", "ccittfax4","ccittrle","deflate","jpeg","jpeg2000","lerc","lerc_deflate","lerc_zstd","lzma","lzw","none","packbits","webp","zstd"]
    assert COMPRESSION_METHOD in allowed_methods, f"Invalid compression method. Choose from {allowed_methods}"

    for file in os.listdir(SOURCE_FOLDER):
        if not file.endswith(".tif"):
            continue
        
        logger.info(f"Compressing {file}")
        
        # Open the source raster file
        source_file_path = os.path.join(SOURCE_FOLDER, file)
        with rasterio.open(source_file_path) as src:
            # Read the file profile
            srcprof = src.profile.copy()
            
            # Update the compression method
            srcprof["compress"] = COMPRESSION_METHOD
        
            # Create and save the compressed raster file
            target_file_path = os.path.join(TARGET_FOLDER, file)
            with rasterio.open(target_file_path, "w", **srcprof) as tgt:
                tgt.write(src.read())

if __name__ == "__main__":
    # Argument and parameter specification
    parser = argparse.ArgumentParser(
        description="Compress TIFF files in a directory")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        default="/Users/nicibe/Desktop/Job/swisstopo_stdl/soil_fribourg/proj-soils/config/config-utilities.yaml")
    args = parser.parse_args()

    # load input parameters from config file
    with open(args.config_file) as fp:
        cfg = yaml.load(
            fp,
            Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Use the provided source folder, target folder, and compression method
    SOURCE_FOLDER = cfg["source_folder"]
    TARGET_FOLDER = cfg["target_folder"]
    COMPRESSION_METHOD = cfg["compression_method"]

    # set up logger
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")
    logger.add(cfg["log_file"], level="INFO")

    logger.info(f"{SOURCE_FOLDER = }")
    logger.info(f"{TARGET_FOLDER = }")    
    logger.info(f"{COMPRESSION_METHOD = }")    

    logger.info("Started Program")
    compress_tiffs(SOURCE_FOLDER, TARGET_FOLDER, COMPRESSION_METHOD)
    logger.info("Ended Program\n")
