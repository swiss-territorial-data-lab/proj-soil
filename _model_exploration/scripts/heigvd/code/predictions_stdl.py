# Copyright (c) OpenMMLab. All rights reserved.

import warnings
warnings.simplefilter("ignore", UserWarning)

from argparse import ArgumentParser
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,50))
import mmcv
import numpy as np

import rasterio

import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv.runner import load_checkpoint
from mmseg.core import get_classes
import cv2
import os.path as osp
import torch
import time

def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('img', help='Image directory')
    parser.add_argument('--out', type=str, default="demo", help='out dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    mmcv.mkdir_or_exist(args.out)
    list_image = os.listdir(args.img)
    len_tot = len(list_image)
    i = 0
    start = time.time()

    config_mmcv = mmcv.Config.fromfile(args.config)
    # build the model from a config file and a checkpoint file

    model = init_segmentor(config_mmcv, checkpoint=None, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = get_classes(args.palette)


    for img in list_image:
        
        file_path_img = os.path.join(args.img, img)
        img = cv2.imread(file_path_img, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
        with rasterio.open(file_path_img, "r") as src:
            profile = src.profile
            profile.update(count=1, compress='lzw')
            if "photometric" in profile:
                del profile["photometric"]
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channels = img.shape
        print('Treatment of : ', file_path_img, ' : shape h, w, c : ', height, ', ', width, ', ', channels)
        result = inference_segmentor(model, img)
        out_path = osp.join(args.out, osp.basename(file_path_img).replace('.tif', '_10cm_PRED.tif'))
        image = np.asarray(result, np.uint8).reshape((height, width)) # height, width
        
        # cv2.imwrite(out_path, image)
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(image, 1)

        print(f"Result is save at {out_path}")
        i+=1
        print(i, 'images sur ', len_tot, ' ', i/len_tot*100, '% accompli.\n')
        end = time.time()
        print(end-start, 'sec ecoulees')
        break



if __name__ == '__main__':
    main()
