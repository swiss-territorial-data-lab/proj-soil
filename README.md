
# GitHub repo for the developement of the Automatic Segmentation of Soils


This repo contains the code for the STDL SeB-project - SeB for "Segmentierung des Bodens", automatic segmentation of soils in english - developed in collaboration with the Canton of Fribourg and the Canton of Vaud. The general description of the project is available on the [STDL website](https://www.stdl.ch/fr/Nos-projets/Generation-automatique-d-%252339%253Bune-carte-a-haute-resolution-des-surfaces-de-pleine-terre.htm) and the technical report on the [tech website](https://tech.stdl.ch/PROJ-SOILS/).


## Hardware requirements

A minimum of 16 GB of RAM and a GPU with at least 5 GB of vRAM are needed to infer with the deep learning (DL) model for segmentation.

To train the model, a minimum of 32 GB of RAM and a GPU with at least 16 GB of vRAM are needed.

## Software Requirements

Install [Docker](https://docs.docker.com/get-docker/) to be able to run the Docker config files that will create the environment with all the dependencies.

Hints to use Docker are given in the [Running Docker section](#running-docker).

## Scripts and Procedure

The `scripts/` folder consists of:

1. Jupyter notebooks that run the pipelines in the correct order using Jupyter's command line interface
2. The `heigvd/` folder that contains the code to run the DL-model of the School of Management and Engineering Vaud (HEIG-VD) and the model itself.
3. The`utilities/` folder that contains the functions that are used in the pipelines

More info on the content of these folders is given in the [Folder Structure](#folder-structure) section.

Here are the procedures performed during the project:

* [Productive Inference](#productive-inference)
* [Finetuning of Your Own Model](#fine-tuning-of-your-own-model)
  * [1. Data Preparation](#1-data-preparation)
  * [2. Training](#2-training)
  * [3. Evaluation](#3-evaluation)

Some previous developments are documentend within the `_model_exploration` folder for documentation purposes linked to the [tech website](https://tech.stdl.ch/PROJ-SOILS/), but not for replication ones. You can skip scouting this folder.

## Productive Inference
On top of the explorative developments documented in the `_model_exploration` folder, a productive pipeline (PP) has been put together to upscale the inferences. It can be run in the `model-heigvd` Docker container. 

From the terminal in the GitHub repository, run the command hereafter to set up the container. If issues arise, please consult the [Running Docker section](#running-docker).

```bash
DOCKER_BUILDKIT=0 docker compose build model-heigvd
```

Then, you can begin to prepare the input data for the inference:

0. (Alternative a) On the [SWISSIMAGE 10 cm website](https://www.swisstopo.admin.ch/en/orthoimage-swissimage-10), select the tiles in your area of interest and of the wanted year. Export all download links and save the generated CSV file containing the links in your `data/<year>` folder. Download SWISSIMAGE 10 cm tiles with wget `wget -i file_with_url_links.csv`

```bash
mkdir data/<year>/swi
cd data/<year>/swi
wget -i ../ch.swisstopo.swissimage-*.csv
cd /proj-soils
```

0. (Alternative b) If the wanted year of SWISSIMAGE 10 cm is no more available, `utilities/wmts_geoquery.py` can be used to download the corresponding mosaic from the [SWISSIMAGE Time Travel service](https://www.swisstopo.admin.ch/en/timetravel-aerial-images), passing the script inputs with `config/config-utilities.yaml`. For the given extent (in vector format) on which to download the raster, on should give the number of pixels allowing the output raster to have the desired resolution: 10000 for a 1km-square vector extent result in an ouptut raster having a 10 cm resolution. 

1. Build a VRT of the downloaded tiles with `utilities/build_vrt.py` and `config-pp.yaml`:

2. Generate a grid of subtiles and create subtiles from the VRT with `utilities/from_bpo/prepare_entire_plans.py`, `utilities/constants.py` and  `config-pp.yaml`. The subtiles will be of suitable size for the model (512 by 512 px) with an overlap of 50 px in both direction. In `utilities/constants.py`, you can change the default size and design of the grid. Each cell of the grid has a unique ID made of the x-coordinate and y-coordinate (in dm) of the upper left corner of 970.2 m square regions, and of the row and column number of the subtile in this region: e.g. 26000000_11979752_20_0. 

Commands for steps 1 and 2:
```bash
docker compose run --rm -v <DIR_DATA>:/proj-soils/data -v <DIR_LOGS>:/proj-soils/logs -v <DIR_CONFIG>:/proj-soils/config model-heigvd python3 /app/scripts/utilities/build_vrt.py --config_file /proj-soils/config/config-pp.yaml

docker compose run --rm -v <DIR_DATA>:/proj-soils/data -v <DIR_LOGS>:/proj-soils/logs -v <DIR_CONFIG>:/proj-soils/config model-heigvd python3 /app/scripts/utilities/from_bpo/prepare_entire_plans.py  --config_file /proj-soils/config/config-pp.yaml
```

The following steps are performed by the script `utilities/productive_pipeline.py` with the config file `/proj-soils/config/config-pp.yaml` and the constant in `utilities/constants.py`. The script: 

4. Infers on the subtiles with `infere_heigvd.py`, that lies in the directory `/ViT-Adapter/segmentation/` within the Docker container. With `utilities/constants.py`, you have the possibility to output the prediction only (`LOGIT=False`) or the predictions and a customed probability index (`LOGIT=True`). Discover more about it in the [technical documentation](https://tech.stdl.ch/PROJ-SOILS/#1022-Targeting-Potential-Errors).
5. Cuts the overlapping border of subtiles with `utilities/cut_border.py`
6. Corrects the subtiles that have artefacts with `utilities/post_processing_embedding.py` considering neighboring predictions. Discover more about it in the [technical documentation](https://tech.stdl.ch/PROJ-SOILS/#1023-Correcting-Artefacts).
7. Mosaics the corrected subtiles `utilities/mosaic_grid.py`

```bash
docker compose run --rm -v <DIR_DATA>:/proj-soils/data -v <DIR_LOGS>:/proj-soils/logs -v <DIR_CONFIG>:/proj-soils/config model-heigvd python3 /app/scripts/utilities/productive_pipeline.py  --config_file /proj-soils/config/config-pp.yaml
```

If you are working with two years on the same area, you may be interested to perform monitoring. To do so, redo steps (0) to (7) with the second year and set `MONITORING=True` in `utilities/constants.py`. A usage example can be performed with the config file `config/config-pp-monitoring.yaml`. Here some more specifics, that you will notice in the config file:

* The same output grid from the scripts `prepare_entire_plan.py` has to be used to have comparable tiles to process between both years. 
* The script `utilities/post_processing_embedding.py` takes the previous corrected prediction, correct the actual ones and compute monitoring changes as defined in the [technical documentation](https://tech.stdl.ch/PROJ-SOILS/#1024-Monitoring-Approach).

```bash
docker compose run --rm -v <DIR_DATA>:/proj-soils/data -v <DIR_LOGS>:/proj-soils/logs -v <DIR_CONFIG>:/proj-soils/config model-heigvd python3 /app/scripts/utilities/build_vrt.py --config_file /proj-soils/config/config-pp-monitoring.yaml

docker compose run --rm -v <DIR_DATA>:/proj-soils/data -v <DIR_LOGS>:/proj-soils/logs -v <DIR_CONFIG>:/proj-soils/config model-heigvd python3 /app/scripts/utilities/from_bpo/prepare_entire_plans.py  --config_file /proj-soils/config/config-pp-monitoring.yaml

docker compose run --rm -v <DIR_DATA>:/proj-soils/data -v <DIR_LOGS>:/proj-soils/logs -v <DIR_CONFIG>:/proj-soils/config model-heigvd python3 /app/scripts/utilities/productive_pipeline.py  --config_file /proj-soils/config/config-pp-monitoring.yaml
```

Afterwards, the resulting tiles containing the grouped transitions can be mosaiced and change count can be computed with `utilities/monitoring_count.py` and `config/config-utilities.yaml`

```bash
docker compose run --rm -v <DIR_DATA>:/proj-soils/data -v <DIR_LOGS>:/proj-soils/logs -v <DIR_CONFIG>:/proj-soils/config model-heigvd python3 /app/scripts/utilities/mosaic_grid.py --config_file /proj-soils/config/config-utilities.yaml

docker compose run --rm -v <DIR_DATA>:/proj-soils/data -v <DIR_LOGS>:/proj-soils/logs -v <DIR_CONFIG>:/proj-soils/config model-heigvd python3 /app/scripts/utilities/monitoring_count.py --config_file /proj-soils/config/config-utilities.yaml
```

## Fine-tuning of Your Own Model

### 1. Data Preparation

The Jupyter notebook `train_prep_fr_vd.ipynb` prepares the data for training. It calls different scripts from the `utilities/` folder and uses the config files `config/train/config-train_fr_vd_gt.yaml` and `config/train/config-train_fr_vd_im.yaml`. The data is saved in the `datasets/` directory. All the needed packages are accessible from the Docker container `general-gis`.

```bash
docker compose up -d general-gis
docker ps
docker exec -it <CONTAINER ID> /bin/bash
jupyter server list
```
Open the link of the Jupyter server. You have to change `<CONNECTION ID>` to `localhost`
Example: `http://<CONNECTION ID>:8888/?token=5b95dd37dc81e3df752d455ff7505bf636798eca8831ea25`

In Jupyter lab, open the Jupyter notebook `train_prep_fr_vd.ipynb`, edit and execute the cells. In the end, the dataset structure is as follows:

```text
dataset/name/
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

Training is conducted in the Docker container `model-heigvd`. It not already done, please run the following command to build the Docker image: 

```bash
DOCKER_BUILDKIT=0 docker compose build model-heigvd
```

The training is done using the `train.py` script, that lies in the directory `/ViT-Adapter/segmentation/` within the Docker container. The only parameter that has to be specified is the path to the config file, which specifies the model, the paths to the datasets, the optimizer, the loss function, and the training parameters. The config file is located at `/proj-soils/scripts/heigvd/model/mask2former_beit_adapter_large_512.py`. The config file corresponding to the training will be saved with the training outputs. During training, the process is logged in the `/proj-soils/data/heig-vd_logs_checkpoints` directory (in log, json, and tensorboard format). Depending on the config file, the checkpoints are also saved in this directory: always the most recent one, and the best one (based on the validation mIoU).

Some hints about adapting `/proj-soils/scripts/heigvd/model/mask2former_beit_adapter_large_512.py`:
* l. 2: how many classes are defined in `proj-soils.py`
* l. 3: sum l. 1 and l. 2
* l. 145: the class definition to use
* l. 146: where are the train, val and test datasets
* l. 212: the checkpoint with wich to work. 
* l. 232-235: dimension `max_iters` and `interval`
* l. 237: define the working dir

```bash
docker compose run --rm -v <DIR_DATA>:/proj-soils/data -v <DIR_SCRIPTS>:/proj-soils/scripts -v <DIR_LOGS>:/proj-soils/logs -v <DIR_CONFIG>:/proj-soils/config model-heigvd python3 /ViT-Adapter/segmentation/train.py /proj-soils/scripts/heigvd/model/mask2former_beit_adapter_large_512.py
```

After training, visualize the results in tensorboard:
```bash
python3 -m tensorboard.main --logdir_spec=my_training:/proj-soils/data/data_train/training
```

### 3. Evaluation

After training, you may also want to evaluate the model on the test dataset. To do so, first make the prediction for the test set (script `/ViT-Adapter/segmentation/infere_heigvd.py` and config file `config/infere/config-infere_test.yaml`). Then, please run the script `utilities/metrics_from_cm_per_tile.py` with the config file `config/config-utilities.yaml`. It will ouptut CSV files with several macro- and micro-metrics. 
This can be performed on the multiple classes predicted by the model or on an aggregated version of the prediction. To do so, you should first aggregate the predicted classes into the [superclasses](https://tech.stdl.ch/PROJ-SOILS/#101-Additional-Ground-Truth). This is done with `utilities/reclassify.py` and `config/config-utilities.py`.

Similarly, the script  `utilities/uncertainty_per_tile.py` allows to study the uncertainty aware accuracy evolution with threshold values on the prediction probatility of the model. Please use the config file `config/config-utilities.yaml` to pass the parameters. 


```bash
docker compose run --rm -v <DIR_DATA>:/proj-soils/data -v <DIR_LOGS>:/proj-soils/logs -v <DIR_CONFIG>:/proj-soils/config model-heigvd python3 /ViT-Adapter/segmentation/infere_heigvd.py --config_file /proj-soils/config/infere/config-infere_heigvd-test.yaml

docker compose run --rm -v <DIR_DATA>:/proj-soils/data -v <DIR_LOGS>:/proj-soils/logs -v <DIR_CONFIG>:/proj-soils/config model-heigvd python3 /app/scripts/utilities/reclassify.py --config_file /proj-soils/config/config-utilities.yaml

docker compose run --rm -v <DIR_DATA>:/proj-soils/data -v <DIR_LOGS>:/proj-soils/logs -v <DIR_CONFIG>:/proj-soils/config model-heigvd python3 /app/scripts/utilities/metrics_from_cm_per_tile.py --config_file /proj-soils/config/config-utilities.yaml

docker compose run --rm -v <DIR_DATA>:/proj-soils/data -v <DIR_LOGS>:/proj-soils/logs -v <DIR_CONFIG>:/proj-soils/config model-heigvd python3 /app/scripts/utilities/uncertainty_per_tile.py --config_file /proj-soils/config/config-utilities.yaml
```

## Running Docker
To be able to run the `DOCKER_BUILDKIT=0 docker compose build model-heigvd` command, please check the presence of `"default-runtime": "nvidia"` in the `/etc/docker/daemon.json` file as given here:

```
{
	"runtimes": {
		"nvidia": {
			"args": [],
			"path": "nvidia-container-runtime"
		}
	},
	"default-runtime": "nvidia"
}
```

If edits were necessary, please restart the Docker of your machine.

## Folder Structure

The folder structure of this project is as follows:

```text
proj-soils
│
├── config
│   │
│   ├── infere
│   │   Configuration files concerning the inference pipeline
│   │
│   ├── train
│   │   Configuration files concerning the training pipeline
│   │
│   └── *.yaml: pipeline and utilities config files
│
└── scripts
    │
    ├── heigvd
    │   ├── code : Content is copied to the Docker container during
    │   │   │   the build process 
    │   │   ├── __init__.py: Include the dataset definition
    │   │   │   (proj_soils.py) for mmsegmentation
    │   │   │
    │   │   ├── proj_soils.py: Dataset definition for mmsegmentation
    │   │   │
    │   │   ├── infere_heigvd.py: Script to call the HEIG-VD model
    │   │   │
    │   │   └── train.py: Script to train the HEIG-VD model
    │   │
    │   └── model
    │       ├── encoder_decoder_mask2former.py
    │       │   Source code for the HEIG-VD model, is mounted in the 
    │       │   Docker container at /ViT-Adapter/segmentation/mmseg_custom/models/segmentors/
    │       │
    │       └── mask2former_beit_adapter_large_512*.py
    │           Config files for the HEIG-VD model (for mmsegmentation)
    │
    ├── _model_exploration: Jupyter notebooks and scripts of the model exploration. 
    │
    ├── utilities
    │   Different scripts that are used in the pipelines
    │
    └── *.* Jupyter notebooks with workflows
```
