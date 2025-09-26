#!/bin/python
# -*- coding: utf-8 -*-

#  wmts-geoquery
#
#      Nils Hamel - nils.hamel@alumni.epfl.ch
#      Huriel Reichel
#      Maria Klonner
#      Copyright (c) 2020-2022 Republic and Canton of Geneva
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.



# You know that ...
import os, sys
import yaml
import shutil
import requests
import re
import math
import io
import numpy
import argparse
import geopandas as gpd  

# That too ...
from PIL import Image
from osgeo import gdal  
from osgeo import osr

import warnings
warnings.filterwarnings("ignore")

from loguru import logger


def read_xml_thing( xml ):

    """
    Overview :

    This function is used to read the WMTS capabilities XML and translates it
    into two arrays to facilitate extraction of data. The first returned array
    contains the XML key path while the second, with corresponding index, gives
    access to the keys content through a dictionary.

    Parameter :

    xml : XML content, as a string

        Content of the fetch XML capabilities file from WMTS service.

    Return :

        The function returns the two composed arrays.
    """

    # Initialise readers
    xmlpath = []
    xmldata = []

    # Accumulator
    acc = ''

    # Current path
    path = ''

    # Current dictionary
    dic = dict()

    # Parsing XML string char by char
    for char in xml:

        # Avoid non-printable char
        if ord( char ) >= 31:

            # Case study
            if char == '<':

                # Check if content was detected between markers
                if acc.strip():

                    # Push content in current dictionary
                    dic["inner"] = acc

                # Clear accumulator
                acc = ''

            elif char == '>':

                # Extract first and last char
                fstchar = acc[+0]
                lstchar = acc[-1]

                # Avoid special markers
                if fstchar != '!' and fstchar != '?':

                    # Case study
                    if fstchar == '/':

                        # Dump current path and associated dictionary
                        xmlpath.append( path )
                        xmldata.append( dic  )

                        # Update current path
                        path = '/'.join(path.split('/')[:-1])

                    elif lstchar == '/':

                        # Update current path
                        path = path + '/' + acc.split(' ')[0]

                        # Pre-process marker content
                        acc = acc.replace( '"', ' ' )
                        acc = acc.replace( '=', ' ' )

                        # Decompose marker content
                        split = re.sub( ' +', ' ', acc.strip() ).split( ' ' )

                        # Push marker content in current dictionary
                        for index in range( 1, len( split ) - 1, 2 ):
                            dic[split[index]] = split[index+1]

                        # Dump current path and associated dictionary
                        xmlpath.append( path )
                        xmldata.append( dic  )

                        # Update current path
                        path = '/'.join(path.split('/')[:-1])

                    else:

                        # Update current path
                        path = path + '/' + acc.split(' ')[0]

                # Clear accumulator
                acc = ''

                # Clear dictionary
                dic = dict()

            else:

                # Accumulate chars
                acc = acc + char

    # Return populated readers
    return xmlpath, xmldata



def detect_layer( xmlpath, xmldata, identifier ):

    """
    Overview :

    This function is responsible of locating the layer information in the XML
    translation. The required layer information are then extracted and returned
    through a dictionary with keys :

        Identifier, Style, TileMatrixSet, ResourceURL

    and their found corresponding values.

    Parameters :

    xmlpath : XML translation keys path

        This array of keys path is the result of this script translation of the
        XML content into an easy to access structure.

    xmldata : XML translation keys content

        This array of dictionaries, working with the xmlpath array index-wise,
        is giving access to each key content.

    identifier : target layer identifier

        This string gives the identifier of the layer from which information are
        extracted.

    Return :

        On success, the composed dictionary is returned. None otherwise.
    """

    # Layer dictionary
    layer = dict()

    # Parsing path
    for index in range( 0, len( xmlpath ) ):

        # Detect and push
        if xmlpath[index] == '/Capabilities/Contents/Layer/ows:Identifier':
            layer["Identifier"] = xmldata[index]["inner"]

        # Detect and push
        if xmlpath[index] == '/Capabilities/Contents/Layer/Style/ows:Identifier':
            layer["Style"] = xmldata[index]["inner"]

        # Detect and push
        if xmlpath[index] == '/Capabilities/Contents/Layer/TileMatrixSetLink/TileMatrixSet':
            layer["TileMatrixSet"] = xmldata[index]["inner"]

        # Detect and push
        if xmlpath[index] == '/Capabilities/Contents/Layer/ResourceURL':
            layer["ResourceURL"] = xmldata[index]["template"]

        # Detect layer path
        if xmlpath[index] == '/Capabilities/Contents/Layer':

            # Detect target layer
            if identifier == layer["Identifier"]:

                # Return populated dictionary
                return layer

    # Layer not found
    return None



def detect_tile_matrix( xmlpath, xmldata, identifier ):

    """
    Overview :

    This function is responsible of reading the WMTS layer matrix set scales
    information. It detects the matrix set associated with the target layer and
    reads all the required information of each scale.

    Parameters :

    xmlpath : XML translation keys path

        This array of keys path is the result of this script translation of the
        XML content into an easy to access structure.

    xmldata : XML translation keys content

        This array of dictionaries, working with the xmlpath array index-wise,
        is giving access to each key content.

    identifier : target tile matrix identifier

        This string has to gives the matrix set identifier associated with the
        layer in which queries need to be performed.

    Return :

        This function returns the scales information through a series of arrays
        were index act as scale id :

            scale id, scale denominator, scale origin x and y, scale tile x and
            y pixel size, matrix width and height

        In case the matrix set is not found, all this array are returned with
        None.
    """

    # Identifier
    detect_ident = None

    # Stack element
    scale_id         = None
    scale_denom      = None
    scale_origin_x   = None
    scale_origin_y   = None
    scale_pixel_x    = None
    scale_pixel_y    = None
    scale_mat_width  = None
    scale_mat_height = None

    # Initialise arrays
    array_scale = []
    array_denum = []
    array_org_x = []
    array_org_y = []
    array_pix_x = []
    array_pix_y = []
    array_mat_x = []
    array_mat_y = []

    # Parsing path
    for index in range( 0, len( xmlpath ) ):

        # Detect and push
        if xmlpath[index] == '/Capabilities/Contents/TileMatrixSet/TileMatrix/ows:Identifier':
            scale_id = int( xmldata[index]["inner"] )

        # Detect and push
        if xmlpath[index] == '/Capabilities/Contents/TileMatrixSet/TileMatrix/ScaleDenominator':
            scale_denom = float( xmldata[index]["inner"] )

        # Detect and push
        if xmlpath[index] == '/Capabilities/Contents/TileMatrixSet/TileMatrix/TopLeftCorner':
            buffer_decomp = xmldata[index]["inner"].split( ' ' )
            scale_origin_x = float( buffer_decomp[0] )
            scale_origin_y = float( buffer_decomp[1] )

        # Detect and push
        if xmlpath[index] == '/Capabilities/Contents/TileMatrixSet/TileMatrix/TileWidth':
            scale_pixel_x = int( xmldata[index]["inner"] )

        # Detect and push
        if xmlpath[index] == '/Capabilities/Contents/TileMatrixSet/TileMatrix/TileHeight':
            scale_pixel_y = int( xmldata[index]["inner"] )

        # Detect and push
        if xmlpath[index] == '/Capabilities/Contents/TileMatrixSet/TileMatrix/MatrixWidth':
            scale_mat_width = int( xmldata[index]["inner"] )

        # Detect and push
        if xmlpath[index] == '/Capabilities/Contents/TileMatrixSet/TileMatrix/MatrixHeight':
            scale_mat_height = int( xmldata[index]["inner"] )

        # Detect tilematrix path
        if xmlpath[index] == '/Capabilities/Contents/TileMatrixSet/TileMatrix':

            # Push scale information on array
            array_scale.append( scale_id         )
            array_denum.append( scale_denom      )
            array_org_x.append( scale_origin_x   )
            array_org_y.append( scale_origin_y   )
            array_pix_x.append( scale_pixel_x    )
            array_pix_y.append( scale_pixel_y    )
            array_mat_x.append( scale_mat_width  )
            array_mat_y.append( scale_mat_height )

        # Detect and push
        if xmlpath[index] == '/Capabilities/Contents/TileMatrixSet/ows:Identifier':
            detect_ident = xmldata[index]["inner"]

        # Detect tilematrixset path
        if xmlpath[index] == '/Capabilities/Contents/TileMatrixSet':

            # Detect target tilematrixset
            if detect_ident == identifier:

                # Return composed arrays
                return array_scale, array_denum, array_org_x, array_org_y, array_pix_x, array_pix_y, array_mat_x, array_mat_y

            # Clear arrays
            array_scale = []
            array_denum = []
            array_org_x = []
            array_org_y = []
            array_pix_x = []
            array_pix_y = []
            array_mat_x = []
            array_mat_y = []        

    # Tilematrixset not detected
    return None, None, None, None, None, None, None, None



def get_tile_by_bounding_box( param_url, param_bbox, param_size, param_output, param_tmp ):

    """
    Overview :

    This function is responsible of translating a tile bounding box and pixel
    width into a _GeoTIFF_ image queried through a WMTS service, for now the
    swisstopo SWISSIMAGE one.

    The function starts by checking the geographical and pixel ratio to ensure
    consistency and equal pixel size in both direction.

    The function then searches the first WMTS scale that overshoot the tile
    GSD. This scale is then used to create the list of WMTS tiles to query to
    cover the full desired tile.

    The WMTS tiles are then fetch using the swisstopo services and put into a
    mosaic. This mosaic is then exported, in a temporary directory, in GeoTIFF
    format. The GeoTIFF geographical information are computed based on the
    WMTS scale value and tile index.

    The exported mosaic is then cropped using GDAL binary (gdalwarp), through a
    system call, to crop and resize it according to the provided bounding box
    and pixel width and height.

    The function clears all temporary elements it created in the temporary
    directory.

    Parameters :

    param_url : Prepare query URL

        This URL is used to perform the tile queries. The scale, tile x and y
        index are replaced for each tile.

    param_bbox : Float array

        Desired tile bounding box : [ x_min, y_min, x_max, y_max ].

    param_size : Integer array

        Pixel width and height of the desired tile : [ w, h ].

    param_output : String

        Tile GeoTIFF exportation path.

    param_tmp : String

        Path of the temporary directory to use.

    Return : Nothing
    """

    # Check if target file is already there
    if os.path.isfile( param_output ):

        # Display message
        print( f'Information : file {os.path.basename(param_output)} already exists - Skipping' )

        # Abort query
        return

    # Check ratio
    if round( ( param_bbox[2] - param_bbox[0] ) / ( param_bbox[3] - param_bbox[1] ), 2 ) != round( param_size[0] / param_size[1], 2 ):

        # Display message
        print( 'Error : inconsistent ratio' )

        # Abort procedure
        sys.exit( 1 )


    # Compute pixel size
    pixel_size = ( param_bbox[2] - param_bbox[0] ) / param_size[0]

    # Initialise optimal scale
    optimal_scale = -1

    # Initialise search
    search_scale = min_scale_service

    # Search optimal scale
    while ( search_scale <= max_scale_service ) and ( optimal_scale < 0 ):

        # Compute scale pixel size
        scale_pixel_size = denominator_service[search_scale] * 0.28e-3

        # Detect optimal scale
        if ( scale_pixel_size <= pixel_size ) or ( search_scale == max_scale_service ):

            # Assign optimal scale
            optimal_scale = search_scale

        # Next scale
        search_scale = search_scale + 1


    # Extract scale specific information
    scale_pixel_size    = denominator_service[optimal_scale] * 0.28e-3
    scale_origin_x      = x_origin_service[optimal_scale]
    scale_origin_y      = y_origin_service[optimal_scale]
    scale_pixel_x       = x_pixel_service[optimal_scale]
    scale_pixel_y       = y_pixel_service[optimal_scale]
    scale_matrix_width  = mat_width_service[optimal_scale]
    scale_matrix_height = mat_height_service[optimal_scale]


    # Compute tiling factors
    factor_x = scale_pixel_size * scale_pixel_x
    factor_y = scale_pixel_size * scale_pixel_y

    # Compute ranges
    parsing_low_x = math.floor( ( + param_bbox[0] - scale_origin_x ) / factor_x )
    parsing_hig_x = math.ceil ( ( + param_bbox[2] - scale_origin_x ) / factor_x )
    parsing_low_y = math.floor( ( - param_bbox[3] + scale_origin_y ) / factor_y )
    parsing_hig_y = math.ceil ( ( - param_bbox[1] + scale_origin_y ) / factor_y )

    # Query index
    query_index = 0

    # Compute composite image resolution
    composite_w = ( parsing_hig_x - parsing_low_x + 1 ) * scale_pixel_x
    composite_h = ( parsing_hig_y - parsing_low_y + 1 ) * scale_pixel_y

    # Initialise final image
    composite_image = Image.new( 'RGB', ( composite_w, composite_h ) )

    # Parsing tiles - longitude
    for tile_x in range( parsing_low_x, parsing_hig_x + 1 ):

        # Parsing tiles - latitude
        for tile_y in range( parsing_low_y, parsing_hig_y + 1 ):

            # Push service url
            tile_url = param_url

            # Push scale in tile url
            tile_url = tile_url.replace( '{TileMatrix}', f'{optimal_scale}' )

            # Push coordinates in tile url
            tile_url = tile_url.replace( '{TileCol}', f'{tile_x}' )
            tile_url = tile_url.replace( '{TileRow}', f'{tile_y}' )

            # Display query URL - In case of need, simply un-comment this line
            print( tile_url )

            # Perform query to service
            query = requests.get( tile_url, allow_redirects=True, verify=False )

            # Check query
            if query.status_code == 200:

                # Read image from query answer
                temporary_image = Image.open( io.BytesIO( query.content ) )

                # Copy image content at relevant location in composite image
                composite_image.paste( temporary_image, ( ( tile_x - parsing_low_x ) * scale_pixel_x, ( tile_y - parsing_low_y ) * scale_pixel_y ) )

                # Free temporary image
                temporary_image.close()

                # Update index
                query_index = query_index + 1

            else:

                # Display message
                print( 'Error : query failed' )

                # Abort procedure
                sys.exit( 1 )

    # Compose temporary image path
    geotiff_temp = os.path.join(DIR_TMP, f"{int(param_bbox[0])}_{int(param_bbox[1])}.tif")

    # Query geographical fram definition
    geotiff_frame = osr.SpatialReference()

    # Select CH1903+ frame
    geotiff_frame.ImportFromEPSG( EPSG_PROJ )

    # Convert PIL image to numpy array for GDAL
    geotiff_byte = numpy.array( composite_image )

    # Create geotiff container
    geotiff_image = gdal.GetDriverByName('GTiff').Create( geotiff_temp, composite_w, composite_h, 3, gdal.GDT_Byte )
 

    # GeoTiff transformation and projection
    geotiff_image.SetGeoTransform( ( scale_origin_x + parsing_low_x * factor_x, scale_pixel_size, 0, scale_origin_y - parsing_low_y * factor_y, 0, -scale_pixel_size ) )
    geotiff_image.SetProjection( geotiff_frame.ExportToWkt() )

    # GeoTiff color layer
    geotiff_image.GetRasterBand(1).WriteArray( geotiff_byte[:,:,0] )
    geotiff_image.GetRasterBand(2).WriteArray( geotiff_byte[:,:,1] )
    geotiff_image.GetRasterBand(3).WriteArray( geotiff_byte[:,:,2] )

    # Purge and clear GeoTiff container
    geotiff_image.FlushCache()
    # geotiff_image = None

    # Final crop to get the desired tile - Not a satisfying solution, should be
    # performed within the script to spare one temporary file

    # geotiff_temp2 = os.path.join(DIR_TMP, f"{int(param_bbox[0])}_{int(param_bbox[1])}_cut.tif")
    # os.system( f"gdalwarp -of GTiff -s_srs epsg:{EPSG_PROJ} -t_srs epsg:{EPSG_PROJ} -te {param_bbox[0]} {param_bbox[1]} {param_bbox[2]} {param_bbox[3]} -ts {param_size[0]} 0 -r cubic {geotiff_temp} {geotiff_temp2}" )
    # Raster = gdal.Open(geotiff_temp, gdal.GA_ReadOnly)
    tmp = gdal.Warp(param_output, geotiff_image, outputBounds=[param_bbox[0], param_bbox[1], param_bbox[2], param_bbox[3]], resampleAlg=gdal.GRA_NearestNeighbour, options=['COMPRESS=DEFLATE'])
    tmp = None
    geotiff_image = None

    # Remove temporary file
    # shutil.move(geotiff_temp2, param_output)
    os.remove( geotiff_temp )



if __name__ == "__main__":
   # Argument and parameter specification
    parser = argparse.ArgumentParser(
        description="The script queries swisstopo WMTS and saves TIF tiles.")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        default="/proj-soils/config/config-utilities.yaml")
    args = parser.parse_args()

    # load input parameters
    with open(args.config_file) as fp:
        cfg = yaml.load(
            fp,
            Loader=yaml.FullLoader)[os.path.basename(__file__)]

    TARGET_FOLDER = cfg["target_folder"]
    URL_WMTS = cfg["url_wmts"]
    EPSG_PROJ = cfg["epsg_proj"]
    LAYER = cfg["layer"]
    TILES = cfg["tiles"]
    WIDTH = cfg["width"]
    TIME = cfg["time"]
    DIR_TMP = cfg["dir_tmp"]
    TMP = cfg["tmp"]
    LOG_FILE = cfg["log_file"]

    logger.info(f"{TARGET_FOLDER = }")   
    logger.info(f"{URL_WMTS = }")   
    logger.info(f"{EPSG_PROJ = }")       
    logger.info(f"{LAYER = }")
    logger.info(f"{TILES = }")      
    logger.info(f"{WIDTH = }") 
    logger.info(f"{TIME = }")    
    logger.info(f"{DIR_TMP = }")    
    logger.info(f"{LOG_FILE = }")

    os.makedirs(os.path.dirname(TARGET_FOLDER), exist_ok=True)

    # This segment of the code is responsible of fetching the WMTS capabilities
    # files using the service URL provided as parameter. It then reads the XML
    # file in order to extract all the necessary information to perform the
    # geographical-based queries.
    #
    # It also prepare the service URL, found in the XML, by replacing the tiles
    # non-related information, such as time and layer style.

    # Query WMTS XML capabilities    
    service = requests.get(URL_WMTS, allow_redirects=True, verify=False )

    # Extract XML content
    service_xml = service.content.decode("utf-8")

    # Read XML content
    service_xml_path, service_xml_data = read_xml_thing( service_xml )

    # Extract layer information
    layer = detect_layer( service_xml_path, service_xml_data, LAYER )

    # Check layer detection
    if layer == None:

        # Display message
        print( 'Error : Unable to find layer in provided service' )

        # Abort script
        sys.exit( 1 )

    # Extract layer information
    layer_scales, denominator_service, x_origin_service, y_origin_service, x_pixel_service, y_pixel_service, mat_width_service, mat_height_service = detect_tile_matrix( service_xml_path, service_xml_data, layer["TileMatrixSet"] )

    # Compose scale limits
    min_scale_service = 0
    max_scale_service = len( layer_scales ) - 1

    # Check matrix set detection
    if layer_scales == None:

        # Display message
        print( 'Error : Unable to locate tile matrix set of layer' )

        # Abort script
        sys.exit( 1 )

    # Assign query URL
    query_url = layer["ResourceURL"]

    # Prepare ressource URL - Replace time value
    if query_url.find( '{Time}' ) != -1:
        query_url = query_url.replace( '{Time}', TIME )

    # Prepare ressource URL - Replace style value
    if query_url.find( '{Style}' ) != -1:
        query_url = query_url.replace( '{Style}', layer["Style"] )

    # Prepare ressource URL - Replace style value
    if query_url.find( '{TileMatrixSet}' ) != -1:
        query_url = query_url.replace( '{TileMatrixSet}', layer["TileMatrixSet"] )

    # This segment of the code reads the provided geographical file through
    # parameter and reads its content, expected to be mono-polygon. Each polygon
    # is read through its bounding box, defining a tile per polygon. The tile
    # bounding box are then sent to the WMTS/WMS converter to obtain the desired
    # geographical tile from a WMTS tile server.

    # Import tile geographical file - Force CH1903+ frame
    print(os.getcwd())
    vector_tile = gpd.read_file( TILES )# .to_crs( EPSG_PROJ )

    # OLD
    for index, tile in vector_tile.iterrows():

        # Extract tile bounding box
        bounding_box = tile['geometry'].bounds

        # Compose bounding box array
        tile_bbox = [ bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3] ]

        # Compute tile geographical size
        tile_geo_w = tile_bbox[2] - tile_bbox[0]
        tile_geo_h = tile_bbox[3] - tile_bbox[1]

        # Assign tile pixel width
        tile_pixel_w = WIDTH

        # Deduce tile pixel height
        tile_pixel_h = round( tile_pixel_w * ( tile_geo_h / tile_geo_w ) )

        # Compose tile size array
        tile_size = [ tile_pixel_w, tile_pixel_h ]

        # Compose tile GeoTiff exportation name - Using low X,Y coordinates, the tile size and pixel size, all rounded to nearest integer
        tile_export = f"{TARGET_FOLDER}{math.floor(bounding_box[0])}-{math.floor(bounding_box[1])}-{math.floor(tile_geo_w)}-{math.floor(tile_geo_h)}-{tile_pixel_w}-{tile_pixel_h}.tif"

        # Query tile from service
        get_tile_by_bounding_box( query_url, tile_bbox, tile_size, tile_export, TMP )

    # Send exit code
    sys.exit( 0 )

