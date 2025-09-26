import os

import numpy as np

import rasterio
from rasterio.merge import merge 

F1 = '/proj-soils/data_vm_bis/aoi_delemont/2017/mosaic_mc'
F2 = '/proj-soils/data_vm_bis/aoi_delemont/2023/mosaic_mc'
BIN = False

list_f1 = ['/proj-soils/data_vm_bis/aoi_delemont/2017/mosaic_mc/region_2591_1245.tif'] #[os.path.abspath(os.path.join(F1, f)) for f in os.listdir(F1)]
list_f2 = ['/proj-soils/data_vm_bis/aoi_delemont/2023/mosaic_mc/region_2591_1245.tif'] #[os.path.abspath(os.path.join(F2, f)) for f in os.listdir(F2)]



if BIN: 
    OUT_TIF_DIFF_BIN = os.path.join(os.path.split(F1)[0],f"diff_{os.path.split(F1)[1]}_{os.path.split(F2)[1]}_bin.tif")
    reclassification_rules = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2}

    mosaic1, out_trans1 = merge(list_f1, output_count=1)
    mosaic2, out_trans2 = merge(list_f2, output_count=1)

    mosaic1_bin = mosaic1 #np.vectorize(reclassification_rules.__getitem__)(mosaic1)
    mosaic2_bin = mosaic2  #np.vectorize(reclassification_rules.__getitem__)(mosaic2)
    binary_diff = mosaic1_bin-mosaic2_bin
    count_diff_binary = np.sum(binary_diff == 0) 

    print(f"In the binary case, {count_diff_binary} pixels over {mosaic1_bin.shape[1]*mosaic1_bin.shape[2]} pixels ({(count_diff_binary/(mosaic1_bin.shape[1]*mosaic1_bin.shape[2])*100)}%) are the same.")

    meta = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': None, 'width': mosaic1.shape[2], 'height': mosaic1.shape[1], 'count': 1, 'crs': 2056, 'transform': out_trans1, 'tiled': False, 'interleave': 'band'}
    with rasterio.open(OUT_TIF_DIFF_BIN, "w", **meta) as dest:
        dest.write(binary_diff)
    print(OUT_TIF_DIFF_BIN)
else: 
    OUT_TIF_DIFF = os.path.join(os.path.split(F1)[0],f"diff_{os.path.split(F1)[1]}_{os.path.split(F2)[1]}_mc.tif")

    mosaic1, out_trans1 = merge(list_f1, output_count=1)
    mosaic2, out_trans2 = merge(list_f2, output_count=1)
    mosaic_diff = (mosaic1-mosaic2)
    count_diff = np.sum(mosaic_diff == 0) 

    print(f"In the multiclass case, {count_diff} pixels over {mosaic1.shape[1]*mosaic1.shape[2]} pixels ({(count_diff/(mosaic1.shape[1]*mosaic1.shape[2])*100)}%) are the same.")

    meta = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': None, 'width': mosaic1.shape[2], 'height': mosaic1.shape[1], 'count': 1, 'crs': 2056, 'transform': out_trans1, 'tiled': False, 'interleave': 'band'}
    with rasterio.open(OUT_TIF_DIFF, "w", **meta) as dest:
        dest.write(mosaic_diff)
    print(OUT_TIF_DIFF)
