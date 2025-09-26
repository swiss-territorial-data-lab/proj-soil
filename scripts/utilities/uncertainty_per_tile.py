import os, sys
import argparse
import yaml
from tqdm import tqdm
import csv
import rasterio
import numpy as np
from sklearn.metrics import confusion_matrix

from loguru import logger

def read_raster(filepath):
    """
    Read a raster filepath and output the raster data.

    Parameters:
    - file_path (str): Path to the folder containing input raster image.

    Returns:
    - raster data (nd.ndarray)
    """

    with rasterio.open(filepath) as src:
        return src.read()  

def uncertainty_aware_accuracy(predicted_labels, confidence_scores, true_labels, labels_id, confidence_threshold=0.7, reclassification_rules=None, to_reclassify=None ):
    """
    Calculate uncertainty-aware accuracy by filtering out low-confidence predictions.
    
    Parameters:
    - pred_probs: np.ndarray of shape (n_samples, n_classes), model's predicted probabilities.
    - true_labels: np.ndarray of shape (n_samples,), true class labels.
    - confidence_threshold: float, minimum confidence to accept a prediction.

    Returns:
    - accuracy: float, uncertainty-aware accuracy.
    - n_confident_predictions: int, number of predictions above the confidence threshold.
    """
    
    # Filter for predictions with confidence above the threshold
    confident_indices = confidence_scores >= confidence_threshold
    confident_predictions = predicted_labels[confident_indices]
    confident_true_labels = true_labels[confident_indices]
    
    # Calculate accuracy only on confident predictions
    if len(confident_predictions) == 0:
        return np.nan, 0, np.zeros((len(labels_id),len(labels_id))) # Return nan accuracy if no predictions are confident enough
    
    if to_reclassify:
        confident_true_labels = np.vectorize(reclassification_rules.__getitem__)(confident_true_labels)
        confident_predictions = np.vectorize(reclassification_rules.__getitem__)(confident_predictions)
   
    accuracy = np.mean(confident_predictions == confident_true_labels)
    n_confident_predictions = len(confident_predictions)
    cm = confusion_matrix(confident_true_labels, confident_predictions, labels=labels_id)

    return accuracy, n_confident_predictions, cm

def uaa_from_cm(DIR_TILES_LABEL, DIR_TILES_PRED, FILE_NAME_ADDENDUM, to_reclassify=False, reclassification_rules=None):
    """
    For each tile, it computes the uncertainty aware accuracy (UAA). It is designed that way to not overflow the 
    memory by accumulating label and pred vectors. 
    In addition, statistics on UAAs per tiles are computed. 

    Parameters:
    - DIR_TILES_LABEL: label rasters.
    - DIR_TILES_PRED: prediction rasters.
    - FILE_NAME_ADDENDUM (str): String to name a family of output files (containings the metrics)
    - to_reclassify (bool) : Condition for reclassifying the prediction classes raster into a raster of superclasses following the
                      reclassification rules.
    - reclassification_rules (dict): rules to map classes into other superclasses.

    Returns:
    - uncertainty.csv: statistics (min, max, median, mean, std, mean ratio of considered pixels) on UAAs per tiles 
    - uaa_per_tile.csv: uncertainty aware accuracy per tile (columns) and per threshold (rows)
    """

    with open(os.path.join(DIR_TILES_PRED,FILE_NAME_ADDENDUM+'uncertainty.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        row_mean = ['threshold', "number of pixels", "uaa from cm", "uaaa mean", "uaa median", "uaa std", "uaa min","uaa max"]
        writer.writerows([row_mean])

    with open(os.path.join(DIR_TILES_PRED,FILE_NAME_ADDENDUM+'uaa_per_tile.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        row = ['threshold', os.listdir(DIR_TILES_LABEL)]
        writer.writerows([row])

    labels = list(set(reclassification_rules.values()))
    thresholds = [0, 0.0125, 0.025, 0.0375, 0.05, 0.0625, 0.075, 0.0875, 0.1, 0.1125, 0.125, 0.1375, 0.15, 0.1625, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] 
    for threshold in thresholds:
        uaa = []
        ratio_conf = [] 
        cm_tot = np.zeros((len(labels),len(labels)))
        for _,tile_name in tqdm(enumerate(os.listdir(DIR_TILES_LABEL)),total=len(os.listdir(DIR_TILES_LABEL)), 
                                desc='Computing uncertainty aware accuracy per tiles...'):
            if tile_name.endswith('.tif'):
                raster1 = read_raster(os.path.join(DIR_TILES_LABEL,tile_name))
                raster2 = read_raster(os.path.join(DIR_TILES_PRED+'/pred',tile_name))
                raster_logit = read_raster(os.path.join(DIR_TILES_PRED+'/score_diff',tile_name))
                assert raster1[0].shape == raster2[0].shape, "Rasters must have the same dimensions"
            else:
                continue

            raster_label = raster1[0].flatten()
            raster_pred = raster2[0].flatten()

            tmp_uaa, tmp_ratio, cm= uncertainty_aware_accuracy(raster_pred, raster_logit[0].flatten(), raster_label, labels, threshold,reclassification_rules,to_reclassify)
            uaa.append(tmp_uaa)
            ratio_conf.append(tmp_ratio)
            cm_tot = cm_tot + cm

        logger.info('Compute statistics for the uncertainty aware accuracy per tiles...')

        uaa_min = min(uaa)
        uaa_max = max(uaa)
        uaa_mean = np.nanmean(uaa)
        uaa_median = np.nanmedian(uaa)
        ratio_conf_sum = np.sum(ratio_conf)
        uaa_std = np.nanstd(uaa)
        uaa_cm = np.sum(np.trace(cm_tot)) / np.sum(cm_tot)

        with open(os.path.join(DIR_TILES_PRED,FILE_NAME_ADDENDUM+'uncertainty.csv'), 'a', newline='') as file:
            writer = csv.writer(file)
            row_mean = [f"{threshold}", f"{ratio_conf_sum}", f"{uaa_cm:.4f}", f"{uaa_mean:.2f}", f"{uaa_median:.2f}", f"{uaa_std:.2f}", f"{uaa_min:.2f}", f"{uaa_max:.2f}"]
            writer.writerows([row_mean])
        
        with open(os.path.join(DIR_TILES_PRED,FILE_NAME_ADDENDUM+'uaa_per_tile.csv'), 'a', newline='') as file:
            writer = csv.writer(file)
            row_mean = [f"{threshold}"] + [f"{uaa_el:.2f}" for uaa_el in uaa]
            writer.writerows([row_mean])

        logger.info(f"Uncertainty aware accuracy: {uaa_mean:.2f} with {ratio_conf_sum} mean pixel ratio at threshold {threshold}")

    return

if __name__ == "__main__":

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script computes uncertainty aware accuracy (UAA) per tile...")
    parser.add_argument(
        "-cfg",
        "--config_file",
        type=str,
        help="Framework configuration file",
        default="/proj-soils/config/config-utilities.yaml",
    )
    args = parser.parse_args()

    # load input parameters
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    DIR_TILES_LABEL = cfg["dir_tiles_label"]
    DIR_TILES_PRED = cfg["dir_tiles_pred"]
    TO_RECLASSIFY = cfg["to_reclassify"]
    RECLASSIFICATION_RULES = cfg["reclassification_rules"]
    FILE_NAME_PREFIX = cfg["file_name_prefix"]
    LOG_FILE = cfg["log_file"] 

    # set up logger
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(LOG_FILE, level="INFO")
    
    logger.info("---------------------")
    logger.info(f"{DIR_TILES_LABEL = }")
    logger.info(f"{DIR_TILES_PRED = }")    
    logger.info(f"{TO_RECLASSIFY = }")
    logger.info(f"{RECLASSIFICATION_RULES = }")      

    FILE_NAME_ADDENDUM = FILE_NAME_PREFIX+'_'
    uaa_from_cm(DIR_TILES_LABEL, DIR_TILES_PRED, FILE_NAME_ADDENDUM, TO_RECLASSIFY, RECLASSIFICATION_RULES)
