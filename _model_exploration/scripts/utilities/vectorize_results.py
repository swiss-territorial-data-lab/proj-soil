import sys
sys.path.append("/Applications/GRASS-8.3.app/Contents/Resources/etc/python")
import grass.script as gscript
from grass.script import core as gcore
from grass_session import Session

import shutil
import os
import subprocess

import argparse
import yaml

from loguru import logger


def grass_logger_wrapper(grass_fun, log_fun=logger.info, *args, **kwargs) -> None:

    """
    Execute a GRASS GIS command and log its output.

    This function wraps the execution of a given GRASS GIS command with the option to log its output.
    
    Args:
        grass_fun (function): The GRASS GIS command execution function.
        log_fun (function, optional): The logging function to use for output. Defaults to logger.info.
        *args: Positional arguments to pass to the GRASS GIS command.
        **kwargs: Keyword arguments to pass to the GRASS GIS command.
    
    Note:
        - This function uses a temporary GRASS GIS location created in the 'tmp' directory.

    Returns:
        None
    """

    p = grass_fun(
        *args,
        **kwargs,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        )

    stdoutdata, stderrdata = p.communicate()
    stderrdata = str(stderrdata).replace(r'\n', '\n').replace(r"\x08", "")
    log_fun("\n"+stderrdata[2:-1])

def vectorize_grass(input_path, output_path) -> None:

    """
    Process and vectorize raster data using a series of GRASS GIS operations.

    This function vectorizes raster files from the input directory and 
    adds some postprocessing using GRASS GIS. The postprocessing consists
    of a removal of polygons smaller than 3 m^2 and adding a Douglas-Peucker
    line simplification with a threshold of 0.1m. The resulting vectorized
    data is saved as Geopackage in the specified output directory.
    
    Args:
        input_path (str): Path to the directory containing input raster files.
        output_path (str): Path to the output directory for saving vectorized data.

    Returns:
        None
    """

    if os.path.exists("tmp/grass_location"):
        shutil.rmtree("tmp/grass_location")

    with Session(
        gisdb="tmp", location="grass_location", 
        create_opts=cfg["crs"]):
    
        for file in os.listdir(input_path):

            if not file.endswith((".tif", ".tiff")):
                continue

            logger.info(f"Processing {file}")
            
            logger.debug("g.gisenv")
            logger.debug(gcore.read_command("g.gisenv", flags="s"))

            # Set projection using the specified EPSG code
            logger.debug("g.proj")
            grass_logger_wrapper(
                log_fun=logger.debug,
                grass_fun=gcore.start_command,
                prog="g.proj",
                flags="c",
                epsg=cfg["crs"][-4:]
                )
            
            # Import the raster file into the GRASS GIS database
            logger.debug("r.in.gdal")
            grass_logger_wrapper(
                log_fun=logger.debug,
                grass_fun=gcore.start_command,
                prog="r.in.gdal",
                input=input_path+file,
                band=1,
                output="rast_IGN",
                flags="o",
                overwrite=True,
                )
            
            # Set the region to match the imported raster
            logger.debug("g.region")
            grass_logger_wrapper(
                log_fun=logger.debug,
                grass_fun=gcore.start_command,
                prog="g.region",
                raster="rast_IGN"
                )
            
            # Convert raster to vector
            logger.debug("r.to.vect")
            grass_logger_wrapper(
                log_fun=logger.debug,
                grass_fun=gcore.start_command,
                prog="r.to.vect",
                flags="s",
                input="rast_IGN",
                output="vectorized",
                type="area",
                column="class",
                overwrite=True
                )
            
            # Clean the vectorized data
            logger.debug("v.clean")
            grass_logger_wrapper(
                log_fun=logger.debug,
                grass_fun=gcore.start_command,
                prog="v.clean",
                input="vectorized",
                output="cleaned",
                type="area",
                overwrite=True,
                tool=["rmarea,rmsa"],
                threshold="3,0"
                )

            # Simplify the vectorized data using Douglas-Peucker method
            logger.debug("v.generalize")
            grass_logger_wrapper(
                log_fun=logger.debug,
                grass_fun=gcore.start_command,
                prog="v.generalize",
                input="cleaned",
                output="generalized",
                type="area",
                overwrite=True,
                method="douglas",
                threshold="0.1"
                )
            
            # Export the simplified vectorized data as Geopackage
            logger.debug("v.out.ogr")
            grass_logger_wrapper(
                log_fun=logger.debug,
                grass_fun=gcore.start_command,
                prog="v.out.ogr",
                type="area",
                input="generalized",
                output=f"{output_path}vectorized_{file.split('.')[0]}.gpkg",
                output_layer=file[7:-6],
                format="GPKG",
                overwrite=True,
                )

if __name__ == "__main__":

    # Argument and parameter specification
    parser = argparse.ArgumentParser(
        description="The script vectorizes all tif-files from a \
            specified folder and adds some postprocessing steps")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        default="proj-soils/config_for_docker/config-utilities.yaml")
    args = parser.parse_args()

    # load input parameters
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]
    
    LOG_FILE = cfg["log_file"]
    INPUT_FOLDER = cfg["input_folder"]
    OUTPUT_FOLDER = cfg["output_folder"]
    
    # set up logger
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")
    logger.add(LOG_FILE, level="INFO")

    logger.info(f"{LOG_FILE = }")
    logger.info(f"{INPUT_FOLDER = }")
    logger.info(f"{OUTPUT_FOLDER = }")


    logger.info("Started Programm")
    vectorize_grass(INPUT_FOLDER, OUTPUT_FOLDER)
    logger.info("Ended Program\n")