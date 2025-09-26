import os
import shutil
from unidecode import unidecode

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
import geopandas as gpd

import rasterio

import fiona
fiona.drvsupport.supported_drivers['WFS'] = 'r'

# Define the folder paths and file paths
tiff_folder = "../../data/horizontal_scratch_mosaics/clipped_to_aois"
beneficiaries_folder = "../../data/for_beneficiaries/folder_for_beneficiaries_test"
aoi_path = "../../data/for_beneficiaries/AOIs_for_beneficiaries"

aoi = gpd.read_file(f"{aoi_path}/AOIs_for_beneficiaries.shp")

wfs_url = "https://geodienste.ch/db/av_0/fra?SERVICE=WFS&REQUEST=GetCapabilities"



# Remove the beneficiaries folder if it already exists
if os.path.exists(beneficiaries_folder):
    shutil.rmtree(beneficiaries_folder)
    
# Create a new beneficiaries folder
os.mkdir(beneficiaries_folder)

# Copy the AOI file to the beneficiaries folder
new_aoi_path = f"{beneficiaries_folder}/AOIs_for_beneficiaries"
if not os.path.exists(new_aoi_path):
    shutil.copytree(aoi_path, new_aoi_path)

# Iterate over the files in the tiff_folder
for i, file in enumerate(os.listdir(tiff_folder)):

    # Skip the file if it is a .DS_Store file or a directory
    if file == ".DS_Store" or os.path.isdir(os.path.join(tiff_folder, file)):
        continue

    file_stem = file.split(".")[0]
    aoi_str = file_stem.split("_")[-1]
    aoi_number = aoi_str[-2:]

    # if aoi_str != "aoi37":
    #     continue
    
    print(f"{file_stem}: {i+1} of {len(os.listdir(tiff_folder))}", end="\n")

    aoi_etiquette = unidecode("_".join(
        aoi
        .query(f"OBJECTID == {aoi_number}")["Etiquette"]
        .values[0]
        .replace(",", "")
        .split(" ")
        ))
    

    
    aoi_folder = f"{beneficiaries_folder}/{aoi_str}_{aoi_etiquette}"
    # Create a folder for each AOI in the beneficiaries folder
    if not os.path.exists(aoi_folder):
        os.mkdir(aoi_folder)

    # Create a folder for each horizontal mosaic in the specific AOI folder
    horizontal_mosaic_folder = os.path.join(
        aoi_folder,
        file_stem)
    if not os.path.exists(horizontal_mosaic_folder):
        os.mkdir(horizontal_mosaic_folder)
    
    # Copy the tiff to the horizontal_mosaic_folder
    if not os.path.exists(os.path.join(horizontal_mosaic_folder, file)):
        shutil.copy(
            os.path.join(tiff_folder, file),
            os.path.join(horizontal_mosaic_folder, file)
        )

    with rasterio.open(os.path.join(tiff_folder, file)) as raster:
        (gpd
            .read_file(
                wfs_url,
                layer="LCSF",
                bbox=[bound for bound in raster.bounds])
            .clip([bound for bound in raster.bounds])
            .dissolve(by="Genre")
            .explode(index_parts=False)
            .reset_index()

    

    # Create an empty GeoDataFrame with specified schema and CRS
    empty_gdf = gpd.GeoDataFrame(geometry=[])
    schema = {"geometry": "Polygon", "properties": {"id": "int", "class": "int", "confidence": "int"}}
    crs = "EPSG:2056"


    # Save the empty GeoDataFrame to a shapefile
    empty_gdf.to_file(
        filename=f"{horizontal_mosaic_folder}/{file_stem}.shp", 
        schema=schema,
        crs=crs)