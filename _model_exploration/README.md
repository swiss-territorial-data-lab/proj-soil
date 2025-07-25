
# GitHub repo for the developement of the Automatic segmentation of soils - SECRET (OFS)!

!!! This repo shall not be disclosed because the OFS code is confidential !!!

This repo contains the code for the STDL SAS-project (SAS for "Segmentation automatique du sol", automatic segmentation of soils in english) developed in collaboration with the Canton of Fribourg and the Canton of Vaud. The general description of the project is available on the [STDL website](https://www.stdl.ch/fr/Nos-projets/Generation-automatique-d-%252339%253Bune-carte-a-haute-resolution-des-surfaces-de-pleine-terre.htm) and the technical report on the [tech website](https://tech.stdl.ch/PROJ-SOILS/).


On top of the explorative developments documented in the folder [Exploration of existing models](##-expoloration-of-existig-models), a productive pipeline (PP) has been put together to upscale the inferences. It can be run in the `model-heigvd` Docker container. 

---

## Hardware requirements

A minimum of 16GB of RAM and a GPU with at least 8GB of vRAM is needed to infer with the HEIG-VD model.
A minimum of 32GB of RAM and a GPU with at least 16GB of vRAM is needed to train the HEIG-VD model.

## Software Requirements

Install [Docker](https://docs.docker.com/get-docker/) to be able to run the Docker config files that will create the environment with all the dependencies.

Hints to use Docker are given in the [Running Docker section](#running-docker).

## Scripts and Procedure

The `scripts/` folder consists of:

2. The `heigvd/` folder that contains the code to run the HEIG-VD DL-model and the model itself.
4. The`utilities/` folder that contains the functions that are used in the pipelines

More info on the content of these folders is given in the [Folder Structure](#folder-structure) section.

## Environment installation

In order to run the model (either for inference or training), additional dependencies have to be installed in the Docker container. The following steps have to be done when first running the container:

```bash
docker compose up -d model-heigvd
docker ps
docker exec -it <CONTAINER ID> /bin/bash
cd /ViT-Adapter/segmentation/ops
python3 setup.py build install
cd /proj-soils
```

## Productive inference
On top of the explorative developments documented in the folder `model_exploration` as well as on the Sections 1 to 9 of the tech website, a productive pipeline (PP) has been put together to upscale the inferences. It can be run in the `model-heigvd` Docker container. 

0. (Alternative a) 0. On the [SWISSIMAGE 10 cm website](https://www.swisstopo.admin.ch/en/orthoimage-swissimage-10), select the tiles in the area of interest and save the generated CSV file containing the download links in your `data` folder. Download SWISSIMAGE 10 cm tiles with wget `wget -i file_with_url_link.csv`
```bash
mkdir data/swi
cd data/swi
wget -i ../ch.swisstopo.swissimage-*.csv
cd /proj-soils
```

0. (Alternative b) If the wanted year of SWISSIMAGE 10 cm is no more available,  `utilities/wmts_geoquery.py` can be used to download tiles from the [SWISSIMAGE Time Travel service](https://www.swisstopo.admin.ch/en/timetravel-aerial-images), passing the script inputs with `config/config-utilities.yaml`. For the given tile extent (in vector format) to download, on should give the number of pixels allowing the output raster to have the desired resolution: 10000 for a 1km-square tile result in an ouptut raster having a 10 cm resolution. 

`python3 scripts/utilities/wmts_geoquery.py --config_file config/config-utilities.yaml`

2. Build a VRT of the downloaded tiles with `utilities/build_vrt.py` and `config-pp.yaml`:
`python3 scripts/utilities/build_vrt.py --config_file config/config-pp.yaml`

3. Generate a grid of subtiles and create subtiles from the VRT with `utilities/from_bpo/prepare_entire_plans.py`, `utilities/constants.py` and  `config-pp.yaml`:
  `python3 scripts/utilities/from_bpo/prepare_entire_plans.py --config_file config/config-pp.yaml`

The following steps can be run at once using `utilities/pp.py` with `/proj-soils/config/config-pp.yaml`, or separetely: 

4. Infer on the subtiles with `heigvd/code/infere_heigvd.py` and `/proj-soils/config/infere/config-pp.yaml`
With `utilities/constants.py`, one has the possibility to output the prediction only (`LOGIT=False`) or the predictions and a confidence index (`LOGIT=True`). Discover more about it in the [technical documentation](https://tech.stdl.ch/PROJ-SOILS/#post_processing)
5. Cut the overlapping border of subtiles with `utilities/cut_border.py` and `/proj-soils/config/config-pp.yaml`
6. Correct the subtiles that have square artefacts with `utilities/post_processing_embedding.py` and `/proj-soils/config/config-pp.yaml`
7. Mosaic the corrected subtiles `utilities/mosaic.py` and `/proj-soils/config/config-pp.yaml`
8. `utilities/build_vrt.py` can be used _a discrétion_ to generate a VRT linking every 1km-mosaic. 

Additionnally, for the same region as previously processed, the another generation of SWISSIMAGE 10 cm can be processed, including a monitoring step (in `constant.py`, `MONITORING=True`) with the previous results. 

## Finetune your own model

### 1. Data preparation

The script `scripts/1-0-train_prep_fr_vd.ipynb` prepares the data for training. It calls different functions from the `utilities/` folder and uses the config file `config/train/config-train_fr_vd_gt.yaml`. The data is saved in the `datasets/` directory. All the needed packages are accessible from the Docker container `general-gis`.

```bash
docker compose up -d general-gis
docker ps
docker exec -it <CONTAINER ID> /bin/bash
jupyter server list
```
Open the link of the listed Jupyter notebook. You have to change the begin of the hyper link, `<CONNECTION ID>` to `localhost`
Example: `http://<CONNECTION ID>:8888/?token=5b95dd37dc81e3df752d455ff7505bf636798eca8831ea25`

The dataset structure is as follows:

```text
dataset/
    ├── test/
    │   ├── ipt/
    │   │   ├── 0.tif
    │   │   ├── 1.tif
    │   │   └── ...
    │   └── tgt/
    │       ├── 0.tif
    │       ├── 1.tif
    │       └── ...
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

The training is done using the `train.py` script, that lies in the directory `/ViT-Adapter/segmentation/` within the Docker container. The only parameter that has to be specified is the path to the config file, which specifies the model, the paths to the datasets, the optimizer, the loss function, and the training parameters. The config file is located at `/proj-soils/scripts/heigvd/model/mask2former_beit_adapter_large_512.py`. The config file corresponding to the training will be saved with the training outputs. During training, the process is logged in the `/proj-soils/data/heig-vd_logs_checkpoints` directory (in log, json, and tensorboard format). Depending on the config file, the checkpoints are also saved in this directory: always the most recent one, and the best one (based on the validation mIoU).

`python3 /ViT-Adapter/segmentation/train.py /proj-soils/scripts/heigvd/model/mask2former_beit_adapter_large_512.py`

After training visualize the results in tensorboard:
```bash


```

## Running Docker

To build and run a specific Docker container with the needed packages, run `docker compose up -d <container>` within the root directory (proj-soils/). This will build the image and run the container.

After the container runs:

1. running containers can be listed using `docker ps`
2. copy the container id of the relevant container and use it for `docker exec -it <container_id> bash`
3. running and stopped containers can be listed using `docker ps -a`
4. stopped containers can be started using `docker start <container_id>`

Note that the `general-gis` container is by default set up to run a Jupyter server at port 8888. Thus, for this container, the command `docker compose up -d general-gis` is enough to expose the server to the local computer. Within the Docker container, one can list the Jupyter server to find the corresponding link `jupyter server list`.

## Folder Structure

The folder structure of this project is as follows:

```text
proj-soils
│
├── config
│   │
│   ├── infere
│   │   Configuration files concerning the inference pipeline for exploration purposes
│   │
│   ├── train
│   │   Configuration files concerning the training pipeline for exploration purposes
│   │
│   └── *.yaml: productive config files
│
└── scripts
    │
    ├── heigvd
    │   ├── code
    │   │   ├── __init__.py: Is copied to the Docker container during
    │   │   │   the build process to include the dataset definition
    │   │   │   (proj_soils.py) for mmsegmentation
    │   │   │
    │   │   ├── proj_soils.py: Dataset definition for mmsegmentation
    │   │   │
    │   │   ├── infere_heigvd.py: Script to call the HEIG-VD model
    │   │   │
    │   │   └── train.py: Script to train the HEIG-VD model, is mounted
    │   │       in the Docker container at /ViT-Adapter/segmentation/
    │   │
    │   └── model
    │       ├── encoder_decoder_mask2former.py
    │       │   Source code for the HEIG-VD model, is mounted in the 
    │       │   Docker container at /ViT-Adapter/segmentation/mmseg_custom/models/segmentors/
    │       │
    │       └── mask2former_beit_adapter_large_512_*.py
    │           Config files for the HEIG-VD model (for mmsegmentation)
    │
    ├── model_exploration: Jupyter notebook of the model exploration. 
    │
    ├── utilities
    │   Different scripts that are used in the pipelines
    │
    └── *.* files of productive workflows
```