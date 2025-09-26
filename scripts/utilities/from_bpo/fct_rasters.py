import geopandas as gpd
import numpy as np
from rasterio.features import shapes
from shapely.geometry import Point, Polygon, shape

import scripts.utilities.constants as cst

from math import ceil

def get_grid_size(tile_size, grid_width=256, grid_height=256, max_dx=0, max_dy=0):
    """Determine the number of grid cells based on the tile size, grid dimension and overlap between tiles.
    All values are in pixels.

    Args:
        tile_size (tuple): tile width and height
        grid_width (int, optional): width of a grid cell. Defaults to 256.
        grid_height (int, optional): height of a grid cell. Defaults to 256.
        max_dx (int, optional): overlap on the width. Defaults to 0.
        max_dy (int, optional): overlap on the height. Defaults to 0.

    Returns:
        number_cells_x: number of grid cells on the width
        number_cells_y: number of grid cells on the height
    """

    tile_width, tile_height = tile_size
    number_cells_x = ceil((tile_width - max_dx)/(grid_width - max_dx))
    number_cells_y = ceil((tile_height - max_dy)/(grid_height - max_dy))

    return number_cells_x, number_cells_y


def get_bbox_origin(bbox_geom):
    """Get the lower xy coorodinates of a bounding box.

    Args:
        bbox_geom (geometry): bounding box

    Returns:
        tuple: lower xy coordinates of the passed geometry
    """

    coords = bbox_geom.exterior.coords.xy
    min_x = min(coords[0])
    min_y = min(coords[1])

    return (min_x, min_y)


def get_east_north(bbox_geom):
    """
    Get the maximum east and north coordinates from a bounding box geometry.

    Args:
        bbox_geom (shapely.geometry.Polygon): The bounding box geometry.

    Returns:
        tuple: A tuple containing the maximum east and north coordinates.
    """

    coords = bbox_geom.exterior.coords.xy
    max_x = max(coords[0])
    max_y = max(coords[1])

    return (max_x, max_y)


def grid_over_tile(tile_size, tile_origin, pixel_size_x, pixel_size_y=None, max_dx=0, max_dy=0, grid_width=256, grid_height=256, crs='EPSG:2056', test_shape = None):
    """Create a grid over a tile and save it in a GeoDataFrame with each row representing a grid cell.

    Args:
        tile_size (tuple): tile width and height
        tile_origin (tuple): tile minimum coordinates
        pixel_size_x (float): size of the pixel in the x direction
        pixel_size_y (float, optional): size of the pixels in the y drection. If None, equals to pixel_size_x. Defaults to None.
        max_dx (int, optional): overlap in the x direction. Defaults to 0.
        max_dy (int, optional): overlap in the y direction. Defaults to 0.
        grid_width (int, optional): number of pixels in the width of one grid cell. Defaults to 256.
        grid_height (int, optional): number of pixels in the height of one grid cell. Defaults to 256.
        crs (str, optional): coordinate reference system. Defaults to 'EPSG:2056'.

    Returns:
        GeoDataFrame: grid cells and their attributes
    """

    min_x, min_y = tile_origin

    number_cells_x, number_cells_y = get_grid_size(tile_size, grid_width, grid_height, max_dx, max_dy)

    # Convert dimensions from pixels to meters
    pixel_size_y = pixel_size_y if pixel_size_y else pixel_size_x
    grid_x_dim = grid_width * pixel_size_x
    grid_y_dim = grid_height * pixel_size_y
    max_dx_dim = max_dx * pixel_size_x
    max_dy_dim = max_dy * pixel_size_y

    # Create grid polygons
    polygons = []
    for x in range(number_cells_x):
        for y in range(number_cells_y):
            
            down_left = (min_x + x * (grid_x_dim - max_dx_dim), min_y + y * (grid_y_dim - max_dy_dim))

            # Fasten the process by not producing every single polygon
            if test_shape and not (test_shape.intersects(Point(down_left))):
                continue

            # Define the coordinates of the polygon vertices
            vertices = [down_left,
                        (min_x + (x + 1) * grid_x_dim - x * max_dx_dim, min_y + y * (grid_y_dim - max_dy_dim)),
                        (min_x + (x + 1) * grid_x_dim - x * max_dx_dim, min_y + (y + 1) * grid_y_dim - y * max_dy_dim),
                        (min_x + x * (grid_x_dim - max_dx_dim), min_y + (y + 1) * grid_y_dim - y * max_dy_dim)]

            # Create a Polygon object
            polygon = Polygon(vertices)
            polygons.append(polygon)

    # Create a GeoDataFrame from the polygons
    grid_gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    grid_gdf['id'] = [f'{round(min_x)}, {round(min_y)}' for min_x, min_y in [get_bbox_origin(poly) for poly in grid_gdf.geometry]]

    block_size = cst.REGION_LENGTH

    min_x, min_y, _, _ = grid_gdf.total_bounds

    cell_width = grid_gdf.bounds.iloc[0, 2] - grid_gdf.bounds.iloc[0, 0]
    cell_height = grid_gdf.bounds.iloc[0, 3] - grid_gdf.bounds.iloc[0, 1]

    step_x = cell_width * (grid_width-max_dx)/grid_width 
    step_y = cell_height * (grid_height-max_dx)/grid_height 

    grid_gdf["centroid_x"] = grid_gdf.centroid.x
    grid_gdf["centroid_y"] = grid_gdf.centroid.y

    grid_gdf["col"] = (((grid_gdf.centroid.x - min_x) % block_size) // step_x).astype(int)
    grid_gdf["row"] = (((min_y - grid_gdf.centroid.y ) % block_size) // step_y).astype(int)

    grid_gdf['minx']= np.rint((grid_gdf.bounds.minx - step_x*grid_gdf.col)*10).astype(int) 
    grid_gdf['maxy']= np.rint((grid_gdf.bounds.maxy + step_y*grid_gdf.row)*10).astype(int)

    grid_gdf['file_name']=grid_gdf.minx.astype(str)+'_'+grid_gdf.maxy.astype(str)+'_'+grid_gdf.row.astype(str)+'_'+grid_gdf.col.astype(str)+str('.tif') 

    return grid_gdf


def no_data_to_polygons(image_band, transform, nodata_value, crs="EPSG:2056"):
    """Convert nodata values in raster (numpy array) to polygons
    cf. https://gis.stackexchange.com/questions/295362/polygonize-raster-file-according-to-band-values

    Args:
        images (DataFrame): image dataframe with an attribute named path

    Returns:
        GeoDataFrame: the polygons of the area with nodata values on the read rasters.
    """

    nodata_polygons = []

    nodata_shapes = list(shapes(image_band, mask=image_band == nodata_value, transform=transform))
    nodata_polygons.extend([shape(geom) for geom, value in nodata_shapes])

    nodata_gdf = gpd.GeoDataFrame({'id_nodata_poly': [i for i in range(len(nodata_polygons))], 'geometry': nodata_polygons}, crs=crs)
    # Remove isolated pixels with the same value as nodata
    nodata_gdf = nodata_gdf[nodata_gdf.area > 10].copy()

    return nodata_gdf


def remove_black_border(image):
    """
	Removes a black border from the input image.

	Args:
	    image (numpy array): The input image to remove the black border from.

	Returns:
	    numpy array: The image with the black border removed.
	"""

    coords = np.argwhere(image)
    if len(image.shape) == 2:
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        resized_image = image[x_min:x_max+1, y_min:y_max+1]
    elif len(image.shape) == 3:
        x_min, y_min, z_min = coords.min(axis=0)
        x_max, y_max, z_max = coords.max(axis=0)
        resized_image = image[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
    else:
        raise ValueError('Image must be 2D or 3D')

    return resized_image