import yaml
import os
import sys
import argparse
import time
from loguru import logger

import rasterio
from rasterio.features import shapes
from rasterio.features import rasterize
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.ops import linemerge, unary_union
from shapely.geometry import MultiLineString, LineString, Point
from shapely.geometry import shape

import fct_misc as misc
import constants as cst


def calculate_angle(v1, v2):
    """
    Calculate the angle between two vectors in degrees.
    In addition, apply a custom filter on the magnitude of the multiplication of the two
    vectors and check for convexity
    
    Parameters:
        v1 (tuple): first 2D vector making the angle
        v2 (tuple): second 2D vector making the angle
    
    Returns:
        angle: angle in degree
        magn: magnitude in meter
        is_convex: True if convex
    """

    dot_product = np.dot(v1, v2)
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)
    magn =  mag_v2*mag_v1
    cos_theta = dot_product / magn
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    is_convex = np.False_
    if magn > cst.MIN_MAGN_NORMS_SQUARE_ANGLES:
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        if cross_product > 0:
            is_convex = np.True_

    return angle, magn, is_convex

def right_angles_in_line(line,count_90_degree):
    """
    Search for right angles in Line geometry.
    Apply a custom filter on the angle, on the magnitude of the multiplication of the two
    vectors and for convexity.
    
    Parameters:
        line: geometry line
        count_90_degree: track count of right angles
    
    Returns:
        count_90_degree: how much convex 90 degree angles are in the line geometry
    """

    points = [point for point in line]

    for i in range(len(points)-2):  # Iterate over consecutive segments
        p1 = points[i]
        p2 = points[i+1]
        p3 = points[i+2]

        # Calculate vectors for two consecutive segments
        v1 = tuple(map(lambda i, j: i - j, p1, p2))
        v2 = tuple(map(lambda i, j: i - j, p2, p3))

        # Compute the angle
        angle, magn, is_convex = calculate_angle(v1, v2)
    
        if (abs(angle - 90) < 1 or abs(angle - 270) < 1) and is_convex>0 and magn > cst.MIN_MAGN_NORMS_SQUARE_ANGLES:
            count_90_degree += 1
    
    if points[0]==points[-1]:
        p1 = points[-2]
        p2 = points[0]
        p3 = points[1]

        # Calculate vectors for two consecutive segments
        v1 = tuple(map(lambda i, j: i - j, p1, p2))
        v2 = tuple(map(lambda i, j: i - j, p2, p3))

        # Compute the angle
        angle, magn, is_convex = calculate_angle(v1, v2)
    
        if (abs(angle - 90) < 1 or abs(angle - 270) < 1) and is_convex>0 and magn > cst.MIN_MAGN_NORMS_SQUARE_ANGLES:
            count_90_degree += 1

    return count_90_degree

def right_angles_in_gdf(geo_df):
    """Detect square angles in polygon geometries."""
    """
    Search for right angles in Polygon geometries of a GeoDataFrame.
    Search only in polygons with a maximum perimeter and with a minimum bounding box fill ratio
    Describes the overall situation: how many right angles, among which how many are convex
    
    Parameters:
        line: geometry line
        count_90_degree: track count of right angles
    
    Returns:
        geo_df: input GeoDataFrame with additional attributes.
    """   
    res_dict = {
        'has_square_angle': [],
        'count': [],
        'convex': []
        }
    for geom in geo_df.geometry:
        values = [False, False, False]  
        if geom.geom_type == 'Polygon':
            if (geom.length < cst.MAX_PERIMETER_ARTEFACTS) & (geom.area/geom.envelope.area > cst.MIN_SQUARENESS):    
                coords = list(geom.exterior.coords)
                square_angles = 0
                square_magn = 0
                square_convex = 0
                coords=coords[:-1] #last coord is repeated. 
                for i in range(len(coords)):
                    v1 = np.array(coords[i ]) - np.array(coords[i-1])
                    v2 = np.array(coords[(i + 1) % len(coords)]) - np.array(coords[i])
                    angle, magn, is_convex = calculate_angle(v1, v2)
                    if abs(angle - 90) < 1 or abs(angle - 270) < 1:
                        square_angles += 1
                    if magn > cst.MIN_MAGN_NORMS_SQUARE_ANGLES:
                        square_magn += 1 # magn max or count ?
                    if is_convex is np.True_:
                        square_convex +=1
                values = [square_angles, square_magn, square_convex]        
        
        for key, val in zip(res_dict.keys(), values):
            res_dict[key].append(val)

    geo_df['length'] = geo_df.geometry.exterior.length 
    geo_df = pd.concat([geo_df, pd.DataFrame(res_dict)], axis=1)
    return geo_df

def main(raster_data, transform, to_reclassify, reclassification_rules): 
    """
    Vecotrize a raster of prediction and correct polygon showing right angles of 
    magnitude greater than cst.SQUARENESS and of magnitude greater than cst.MAGN 
    with the neighboring class. 

    Parameters:
        raster_data (np.array): raster of prediction to be corrected
        out_trans (): transform information of the raster_data 
        to_reclassify (bool): are the classes to be reclassified into superclasses?
        reclassification_rules (dict): how to reclassify classes into superclass, if to_reclassify is true
    
    Returns:
        None
    """


    if to_reclassify:
        raster_data = np.vectorize(reclassification_rules.__getitem__)(raster_data)

    pred_column = "DN"
    gdf = misc.raster2gdf(raster_data.astype('int32'), transform, pred_column)

    gdf = right_angles_in_gdf(gdf)

    
    gdf['bbox_fill_ratio']=gdf.area/gdf.geometry.envelope.area
    gdf['length'] = gdf.geometry.length
    selected_polygons = gdf[(gdf['convex'] > 0) & (gdf['convex'] < 5) & 
                            (gdf["length"] < cst.MAX_PERIMETER_ARTEFACTS) & 
                            (gdf['bbox_fill_ratio'] > cst.MIN_SQUARENESS)].copy()
    selected_polygons.drop(columns=['convex'], inplace=True)
    # selected_polygons.to_file('/proj-soils/data_vm/T2/2020/mix_corr_mc/sel4.gpkg')
    # gdf.to_file('/proj-soils/data_vm/T2/2020/mix_corr_mc/gdf4.gpkg')

    # Initialize a new column for updated attributes
    corr_pred_column = "DN_updated"
    gdf[corr_pred_column] = gdf[pred_column]
    # gdf.drop(columns=['convex'], inplace=True)

    # Find neighbors using spatial join
    neighbors = gpd.sjoin(gdf, gdf, predicate='touches', how='left')
    # neighbors.to_file('/proj-soils/data_vm/T2/2020/mix_corr_mc/nei4.gpkg')
    
    # Iterate through selected polygons
    for poly in selected_polygons.itertuples():
        idx = poly[0]
        # Get neighboring polygons   
        neighbor_indices = neighbors[neighbors.index == idx].index_right
        
        # Calculate the length of the shared boundary with each neighbor
        shared_lengths = {}
        shared_angle = {}

        for neighbor_idx in neighbor_indices:

            if np.isnan(neighbor_idx):
                continue
            neighbor = gdf.loc[neighbor_idx]
            shared_boundary = poly.geometry.intersection(neighbor.geometry)     
            # Only consider neighbors with a valid shared boundary
            if isinstance(shared_boundary, LineString) or shared_boundary.length > 0:
                shared_lengths[neighbor_idx] = shared_boundary.length

                if shared_boundary.geom_type not in ['LineString', 'MultiLineString','GeometryCollection']:
                    return 0

                if (shared_boundary.geom_type == 'GeometryCollection'):
                    filtered_geoms = [el for el in shared_boundary.geoms if not isinstance(el, Point)]
                    shared_boundary = unary_union(filtered_geoms)

                s = gpd.GeoSeries(shared_boundary)
                s_merged = s if s.geometry[0].geom_type == 'LineString' else s.apply(linemerge)

                count_90_degree = 0

                if s_merged.geometry.geom_type[0]  == 'MultiLineString':
                    # convert a MultiLineString in a list of LineStrings and then in a list of list of coordinates
                    s_list = list(list(s_merged[i].geoms) for i in range(len(s_merged)))
                    lines = list(list(s_list[i][j].coords) for i in range(len(s_list)) for j in range(len(s_list[i])))

                elif s_merged.geometry.geom_type[0] == 'LineString':
                    # convert a LineString in a list of list of coordinates
                    lines = [list(s_merged[0].coords)]

                for one_line in lines:
                    count_90_degree = right_angles_in_line(one_line, count_90_degree)
                            
                shared_angle[neighbor_idx] = count_90_degree

        # If there are neighbors, find the one with the longest shared boundary
        if shared_angle:
            max_count=max(shared_angle.values())
            if max_count > 0:
                max_keys=[key for key, value in shared_angle.items() if value == max_count]
                selected_lengths = {key: shared_lengths[key] for key in max_keys}
                best_neighbor_idx = max(selected_lengths, key=selected_lengths.get)    
                gdf.at[idx, corr_pred_column] = gdf.at[best_neighbor_idx, pred_column]
    # gdf.to_file('/proj-soils/data_vm/T2/2020/mix_corr_mc/corr2.gpkg')
    # Define output raster parameters
    out_shape = (round((gdf.total_bounds[3]-gdf.total_bounds[1])/0.1), round((gdf.total_bounds[2]-gdf.total_bounds[0])/0.1))
    transform = rasterio.transform.from_bounds(*gdf.total_bounds, out_shape[1], out_shape[0])

    # Rasterize the GeoDataFrame
    raster = misc.gdf2raster(gdf, gdf.columns.get_loc(corr_pred_column)+1, out_shape, transform)
   
    return raster


if __name__ == "__main__":

    # Argument and parameter specification
    parser = argparse.ArgumentParser(
        description="The script corrects the artefacts in the predictions per individual tile.")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        default="/proj-soils/config/config-utilities.yaml")
    args = parser.parse_args()

    # load input parameters
    with open(args.config_file) as fp:
        cfg = yaml.load(
            fp,
            Loader=yaml.FullLoader)[os.path.basename(__file__)]

    TO_RECLASSIFY = cfg["to_reclassify"]
    RECLASSIFICATION_RULES = cfg["reclassification_rules"]
    INPUT_FOLDER = cfg["input_folder"]
    OUTPUT_FOLDER = cfg["output_folder"]
    LOG_FILE = cfg["log_file"]

    # set up logger
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(LOG_FILE, level="INFO")
 
    logger.info(f"{TO_RECLASSIFY = }")
    logger.info(f"{RECLASSIFICATION_RULES = }")      
    logger.info(f"{INPUT_FOLDER = }") 
    logger.info(f"{OUTPUT_FOLDER = }")

    tic = time.time()
    logger.info("Started Programm")

    for file in os.listdir(INPUT_FOLDER):
        if not file.endswith(("tif", "tiff")):
            continue

        raster_path = os.path.join(INPUT_FOLDER, file)
        with rasterio.open(raster_path) as src:
            src_data = src.read(1)
            transform = src.transform
            src_corr = main(src_data, transform, TO_RECLASSIFY, RECLASSIFICATION_RULES)

            out_profile = src.profile.copy()
            with rasterio.open(os.path.join(OUTPUT_FOLDER,file), "w", **out_profile) as dest:
                dest.write(src_corr,1)
    
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")
