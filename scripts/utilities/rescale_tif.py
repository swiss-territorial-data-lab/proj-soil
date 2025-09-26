import rasterio
import numpy as np

import yaml
import os
import shutil
import sys
import argparse

from loguru import logger

import warnings
warnings.filterwarnings('ignore')


def rescale_tif(TIFF_FOLDER, TARGET_RES, OUT_FOLDER) -> None:
    """
    Rescale TIFF files in the specified folder to a target resolution and save them in the output folder.
    
    This function rescales input TIFF files to a specified target resolution using nearest-neighbour interpolation.
    The rescaled files are then saved in the output folder with updated metadata.

    Parameters:
        TIFF_FOLDER (str): Path to the folder containing input TIFF files.
        TARGET_RES (float): Target resolution in the same unit as the input raster.
        OUT_FOLDER (str): Path to the output folder to save the rescaled TIFF files.

    Returns:
        None
    """

    if not isinstance(TIFF_FOLDER, list):
        TIFF_FOLDER = [TIFF_FOLDER]
    if not isinstance(TARGET_RES, list):
        TARGET_RES = [TARGET_RES]
    if not isinstance(OUT_FOLDER, list):
        OUT_FOLDER = [OUT_FOLDER]
    
    
    for CURRENT_TIFF_FOLDER, CURRENT_TARGET_RES, CURRENT_OUT_FOLDER in zip(TIFF_FOLDER, TARGET_RES, OUT_FOLDER):
        
        for file in os.listdir(CURRENT_TIFF_FOLDER):

            if not file.endswith((".tif", ".tiff")):
                continue

            with rasterio.open(os.path.join(CURRENT_TIFF_FOLDER, file)) as src:

                if round(src.res[0], 2) == CURRENT_TARGET_RES and round(src.res[1], 2) == CURRENT_TARGET_RES:
                    logger.info(f"{file} is already at target resolution")
                    try:
                        shutil.copyfile(os.path.join(CURRENT_TIFF_FOLDER, file), os.path.join(CURRENT_OUT_FOLDER, file))
                    except shutil.SameFileError:
                        pass
                    continue
                
                logger.info(f"Rescaling {file}")    
                
                tgt_height = int(src.height * (src.res[0] / CURRENT_TARGET_RES))
                tgt_width = int(src.width * (src.res[0] / CURRENT_TARGET_RES))

                arr_interp = src.read(
                    out_shape=(
                        src.count,
                        tgt_height,
                        tgt_width
                    ),
                    resampling=rasterio.enums.Resampling.nearest
                )
        # # Create an empty array for the downsampled image
        #         data = src.read(1)  # Read first band
        #         transform = src.transform
        #         nodata = src.nodata
        #         arr_interp = np.zeros((tgt_height, tgt_width), dtype=data.dtype)
        #         for i in range(tgt_height):
        #             for j in range(tgt_width):
        #                 row = i * 10 + 10 // 2
        #                 col = j * 10 + 10 // 2

        #                 # Check bounds
        #                 if row < data.shape[0] and col < data.shape[1]:
        #                     arr_interp[i, j] = data[row, col]
        #                 else:
        #                     arr_interp[i, j] = nodata if nodata is not None else 0
                        
                transform_interp = src.transform * src.transform.scale(
                    (src.width / arr_interp.shape[-1]),
                    (src.height / arr_interp.shape[-2])
                )

                out_meta = src.meta.copy()
            
            # Update metadata for the rescaled image
            out_meta.update({
                "driver": "Gtiff",
                "height": tgt_height,
                "width": tgt_width,
                "transform": transform_interp
            })

            with rasterio.open(os.path.join(CURRENT_OUT_FOLDER, file), "w", **out_meta) as tgt:
                tgt.write(arr_interp)


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
    OUT_FOLDER = cfg["out_folder"]
    TARGET_RES = cfg["target_res"]
    LOG_FILE = cfg["log_file"]

    # set up logger
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")
    logger.add(LOG_FILE, level="INFO")

    logger.info(f"{TIFF_FOLDER = }")
    logger.info(f"{OUT_FOLDER = }")
    logger.info(f"{TARGET_RES = }")
    logger.info(f"{LOG_FILE = }")

    os.makedirs(OUT_FOLDER, exist_ok=True)

    logger.info("Started Programm")
    rescale_tif(TIFF_FOLDER, TARGET_RES, OUT_FOLDER)
    logger.info("Ended Program\n")