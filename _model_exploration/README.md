
# Automatic segmentation of soils

This folder contains the code for the first phase of the STDL SAS-project, developed in collaboration with the Canton of Fribourg. Sections 1 to 9 of the technical report on the [Tech Website](https://tech.stdl.ch/PROJ-SOILS/) correspond to this phase. 

Disclaimer:  Path maintenance, data availability and reproducibility are not guaranteed.

---

## Hardware requirements

A minimum of 16GB of RAM and a GPU with at least 8GB of vRAM is needed to infer with the HEIG-VD model.
A minimum of 32GB of RAM and a GPU with at least 16GB of vRAM is needed to train the HEIG-VD model.

## Software Requirements

Install [Docker](https://docs.docker.com/get-docker/) to be able to run the Docker config files that will create the environment with all the dependencies.

Hints to use Docker are given in the [Running Docker section](#running-docker).

## Scripts and Procedure

The `scripts/` folder consists of:

1. Jupyter notebooks that run the pipelines in the correct order using Jupyter's command line interface (starting cells with a '!')
2. The `heigvd/` folder that contains the code to run the DL-model of the Institute of Territorial Engineering (INSIT) at the School of Engineering and Management (HEIG-VD) and the model itself.
3. The`utilities/` folder that contains the functions that are used in the pipelines

More info on the content of these folders is given in the [Folder Structure](#folder-structure) section.

Here is the ordered procedures performed during the project:

* [Evaluation Pipeline](#evaluation-pipeline)
* [Training Pipeline](#training-pipeline)
  * [1. Data Preparation](#1-data-preparation)
  * [2. Training](#2-training)
* [Inference of HEIG-VD model](#inference-of-heig-vd-model)
  * [1. Preparation](#1-preparation)
  * [2. Inference](#2-inference)
  * [3. Post-processing](#3-post-processing)

Please find here the model weights: https://sftpgo.stdl.ch/web/client/pubshares/2HvZAv4VegLzxmXSPbmUeb/browse. The models correspond to the ones evaluated in Figure 19 of the [technical documentation](https://tech.stdl.ch/PROJ-SOILS/#61-evaluation).

## Evaluation Pipeline

The performance of 3 sets of models can be evaluated: 6 models of the french National Institute of Geographic and Forest Information (IGN), 1 model of the Institute of Territorial Engineering (INSIT) at the School of Engineering and Management (HEIG-VD), and 1 model of the Federal Statistical Office (OFS). All the needed packages are accessible from the Docker container `general-gis`.

Use Jupyter notebooks, file name starting with '0-', to run the pipelines in the correct order using Jupyter's commanline interface (starting cells with a '!').
Source and target folders need to be given in the first cell.

1. Preparation
    * 0-0-eval_prep_gt.ipynb  
    * 0-0-eval_prep_heigvd.ipynb
    * 0-0-eval_prep_IGN.ipynb
    * 0-0-eval_prep_OFS.ipynb
2. Evaluation
    * 0-1-eval_calculate_metrics.ipynb
3. Visualization
    * 0-2-visualizations.iypnb

## Training Pipeline

The training pipeline consists of 2 steps:

### 1. Data preparation

The script `1-0-train_prep*.ipynb` prepares the data for training. It calls different functions from the `utilities/` folder and uses config files from the `config/train/` folder. The data is saved in the `datasets/` directory. All the needed packages are accessible from the Docker container `general-gis`.

Note that the split into training and validation data is done randomly in the script `random_split.py`. The random seed 6 was found to be the one that distributes the classes most evenly. However, afterwards, some ID's were manually redistributed to create an even more balanced split.

The dataset structure is as follows:

```text
dataset/
    ├── train/
    │   ├── ipt/
    │   │   ├── 0.tif
    │   │   ├── 1.tif
    │   │   └── ...
    │   └── tgt/
    │       ├── 0.tif
    │       ├── 1.tif
    │       └── ...
    └── val/
        ├── ipt/
        │   ├── 0.tif
        │   ├── 1.tif
        │   └── ...
        └── tgt/
        ├── 0.tif
        ├── 1.tif
        └── ...
```

### 2. Training

Training is conducted in the Docker container `model-heigvd`.

Before training, additional dependencies have to be installed. The following steps have to be done when first running the container:

```bash
cd /ViT-Adapter/segmentation/ops
python3 setup.py build install
```

The training is done using the `train.py` script, that lies in the directory `/ViT-Adapter/segmentation/` within the Docker container. The only parameter that has to be specified is the path to the config file, which specifies  the model, the paths to the datasets, the optimizer, the loss function, and the training parameters. The config file is located at `/proj-soils/scripts/heigvd/model/mask2former_beit_adapter_large_512_*.py`. During training, the process is logged in the `/proj-soils/data/heig-vd_logs_checkpoints` directory (in log, json, and tensorboard format). Depending on the config file, the checkpoints are also saved in this directory: always the most recent one, and the best one (based on the validation mIoU).

## Inference of HEIG-VD model

All necessary configurations are stored in `/proj-soils/config/config-infere_heig-vd.yaml`.

### 1. Preparation

1. **Clip to AOI**: The script `clip_tiffs.py` clipps a geotiff by a specified geopackage.

2. **Convert to RGB**: The script `rgbi2rgb.py` converts a 4-band geotiff to a 3-band geotiff. This step isn't necessary, if SWISSIMAGE10cm imagery is used as input. Note that the checkpoint `M2F_ViTlarge_best_mIoU_iter_160000.pth` has been trained on [FLAIR-1](https://ignf.github.io/FLAIR/#FLAIR1) imagery.

3. (Optional) **Rescale**: If experiments regarding the input resolution are conducted, the script `rescale_tiffs.py` can used to rescale the input images.

### 2. Inference

Inference of the HEIG-VD model is conducted in the `model-heigvd` container, using the script located at `/proj-soils/scripts/heigvd/code/infere_heigvd.py`. All the parameters are specified in the config file. The output is saved in the specified directory. Note that multiple input, output, side_length, and stride values can be stated. If more than one input directory is given, it will loop over the specified directories, using the output, stride, and side_length parameters at the same index as the input directory.

### 3. Post-processing

As the HEIG-VD model has been trained on the French FLAIR-1 classes, the output of the original checkpoint `M2F_ViTlarge_best_mIoU_iter_160000.pth` has to be reclassified to our own classes. This is done using the script reclassify.py. The parameters are specified in the config file `config-infere_heigvd-orig.yaml`.

## Running Docker

To build and run a specific Docker container with the needed packages, run `docker compose up -d <container>` within the root directory (proj-soils/). This will build the image and run the container.

After the container runs:

1. running containers can be listed using `docker ps`
2. copy the container id of the relevant container and use it for `docker exec -it <container_id> bash`

Note that the `general-gis` container is by default set up to run a Jupyter server at port 8888. Thus, for this container, the command `docker compose up -d general-gis` is enough to expose the server to the local computer. Within the Docker container, one can list the Jupyter server to find the corresponding link `jupyter server list`.
<!-- markdownlint-configure-file { "MD051": false} -->

## Folder Structure

The folder structure of this project is as follows:

```text
proj-soils
│
├── config
│   ├── eval
│   │   Configuration files concerning the evaluation pipeline
│   │
│   ├── infere
│   │   Configuration files concerning the inference pipeline
│   │
│   └── train
│       Configuration files concerning the training pipeline
│
└── scripts
    ├── Jupyter notebooks that run the pipelines in the correct order
    │   using Jupyter's command line interface (starting cells with a '!') 
    │
    ├── heigvd
    │   ├── code
    │   │   ├── __init__.py: Is copied to the Docker container during
    │   │   │   the build process to include the dataset definition
    │   │   │   (proj_soils.py) for mmsegmentation
    │   │   │
    │   │   ├── proj_soils.py: Dataset definition for mmsegmentation
    │   │   │
    │   │   ├── infere_heigvd.py: Script to call the HEIG-VD model
    │   │   │
    │   │   ├── infere_heigvd.ipynb: Jupyter notebook to call the HEIG-VD for debug purposes
    │   │   │
    │   │   └── train.py: Script to train the HEIG-VD model, is mounted
    │   │       in the Docker container at /ViT-Adapter/segmentation/
    │   │
    │   └── model
    │       ├── encoder_decoder_mask2former.py
    │       │   Source code for the HEIG-VD model, is mounted in the 
    │       │   Docker container at /ViT-Adapter/segmentation/mmseg_custom/models/segmentors/
    │       │
    │       └── mask2former_beit_adapter_large_512_160k_proj-soils_12class_*.py
    │           Config files for the HEIG-VD model (for mmsegmentation)
    │
    ├── prepare_digitization
    │   ├── folderstructure4beneficiaries.py
    │   │   Script to prepare the folder structure for the digitization
    │   │
    │   └── mosaic_with_OTB.ipynb: Jupyter notebook to horizontally
    │       mosaick tiff files using Orfeo Toolbox
    │
    └── utilities
        Different scripts that are used in the pipelines
```
