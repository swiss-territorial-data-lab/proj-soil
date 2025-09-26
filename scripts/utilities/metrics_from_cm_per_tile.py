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
    Read the first band from a raster filepath and output the raster data.

    Parameters:
    - file_path (str): Path to the folder containing input raster image.

    Returns:
    - raster data (nd.ndarray)
    """

    with rasterio.open(filepath) as src:
        return src.read(1)  

def compute_mcc(conf_matrix):
    """
    Compute Matthew's correlation coefficient from a confusion matrix. 

    Parameters:
    - conf_matrix (np.ndarray): confusion matrix values (row-label, col-pred)

    Returns:
    - numerator/denominator: of the Mattew's correlation coefficient
    """

    # Sum of the diagonal (True Positives across all classes)
    C = np.trace(conf_matrix)
    
    # Total sum of the confusion matrix
    S = np.sum(conf_matrix)
    
    # Sum of each row (actual class counts)
    row_sums = np.sum(conf_matrix, axis=1)
    
    # Sum of each column (predicted class counts)
    col_sums = np.sum(conf_matrix, axis=0)
    
    # Compute MCC
    numerator = (C * S) - np.sum(row_sums * col_sums)
    denominator = np.sqrt((S**2 - np.sum(row_sums**2)) * (S**2 - np.sum(col_sums**2)))
    
    # Avoid division by zero
    return numerator / denominator if denominator != 0 else 0.0

def metrics_from_cm(DIR_TILES_LABEL, DIR_TILES_PRED, FILE_NAME_ADDENDUM, dict_classes, AREA_TYPE, AREA_TYPE_FILE=None):
    """
    Firstly, compute the confusion matrix of a label dataset and corresponding predictions. The computations are made to not 
    overflow the memory, this means that smaller confusion matrices are accumulated. 
    Secondly, compute metrics - common ones (recall, precision, F1-score) and the Mattew's correlation coefficient. 
    In addition, the metrics can be computed by subset (e.g. sub-area) of the dataset. 


    Parameters:
    - DIR_TILES_LABEL (str): Path to the folder containing label rasters.
    - DIR_TILES_PRED (str): Path to the folder containing prediction rasters.
    - FILE_NAME_ADDENDUM (str): String to name a family of output file (containings the metrics)
    - dict_classes (dict): dictionnary linking raster values and string class labels. 
    - AREA_TYPE (str): sub-area on which to do the computation
    - AREA_TYPE_FILE (str): Path to the YAML file containing the sub-area information. 
        For each key in the YAML, the correspondig tile ID are listed.

    Returns:
    - FILE_NAME_ADDENDUM_cm.csv: CSV file containing the confusion matrix
    - FILE_NAME_ADDENDUM_cm_perc.csv: CSV file containing the confusion matrix in percentage of the dataset 
    - FILE_NAME_ADDENDUM_metrics.csv: CSV file containing the metric scores 
    """

    n_classes = len(dict_classes)
    cm = np.zeros((n_classes,n_classes))

    if AREA_TYPE: 
        with open(AREA_TYPE_FILE, 'r') as file:
            area_type = yaml.load(file, Loader=yaml.FullLoader)
            ids_in_area = area_type[AREA_TYPE]
    
    with open(os.path.join(DIR_TILES_PRED,FILE_NAME_ADDENDUM+'iou_per_tile.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        row = ['tile_name']
        writer.writerow(row)
    
    for _,tile_name in tqdm(enumerate(os.listdir(DIR_TILES_LABEL)),total=len(os.listdir(DIR_TILES_LABEL)), desc='Computing confusion matrix from tiles...'):
        if tile_name.endswith('.tif') and (not AREA_TYPE or int(tile_name.split("-")[2].replace(".tif","")) in ids_in_area):
            raster1 = read_raster(os.path.join(DIR_TILES_LABEL,tile_name))
            raster2 = read_raster(os.path.join(DIR_TILES_PRED,tile_name))
        else:
            continue

        # Ensure both rasters have the same shape
        assert raster1.shape == raster2.shape, "Rasters must have the same dimensions"

        # Compute confusion matrix
        labels = np.arange(0, n_classes)
        # cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_temp = confusion_matrix(raster1.flatten(), raster2.flatten(), labels=labels)
        cm = cm + cm_temp

        if False:
            cm_temp = cm_temp.astype('float')
            iou = np.divide(np.diag(cm_temp), (np.sum(cm_temp, axis=1) + np.sum(cm_temp, axis=0) - np.diag(cm_temp)), 
                out=np.zeros_like(np.diag(cm_temp)), where=(np.sum(cm_temp, axis=1) + np.sum(cm_temp, axis=0) - np.diag(cm_temp))!=0)
            cond_min = iou[2]>0
            cond_max = iou[2]<0.5
            if cond_min.any() and cond_max.any():
                with open(os.path.join(DIR_TILES_PRED,FILE_NAME_ADDENDUM+'iou_per_tile.csv'), 'a', newline='') as file:
                    writer = csv.writer(file)
                    row = [tile_name]
                    writer.writerow(row)
  
    # Save the confusion matrix
    np.savetxt(os.path.join(DIR_TILES_PRED,FILE_NAME_ADDENDUM+'cm.csv'), cm, delimiter=",")
    cm_perc = np.round(np.divide(cm, sum(sum(cm)),out=np.zeros_like(cm),where=sum(sum(cm))!=0)*100,2)
    with open(os.path.join(DIR_TILES_PRED,FILE_NAME_ADDENDUM+'cm_perc.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        title= DICT_CLASSES.values()
        writer.writerows([title])
        for id in np.arange(0, n_classes):
            writer.writerows([cm_perc[id]])
    # cm = np.genfromtxt(os.path.join(DIR_CLS1,'cm3.csv'), delimiter=',')

    logger.info('Compute metrics')

    accuracy = np.trace(cm) / np.sum(cm)

    tp_per_class = np.diag(cm)
    labels_per_class = np.sum(cm, axis=1)
    detections_per_class = np.sum(cm, axis=0)
    
    # Precision, recall, F1-score and IoU per class
    precision = np.divide(tp_per_class , detections_per_class, 
                        out=np.zeros_like(tp_per_class), where=detections_per_class!=0)
    recall =  np.divide(tp_per_class, labels_per_class, 
                        out=np.zeros_like(tp_per_class), where=labels_per_class!=0)
    f1_score = 2 *  np.divide(precision * recall, (precision + recall),
                            out=np.zeros_like(precision * recall), where=(precision + recall)!=0)
    iou = np.divide(tp_per_class, (detections_per_class + labels_per_class - tp_per_class), 
                    out=np.zeros_like(tp_per_class), where=(detections_per_class + labels_per_class - tp_per_class)!=0)

    # Global metrics
    macro_precision = np.mean(precision[1:])
    macro_recall = np.mean(recall[1:])
    macro_f1 = np.mean(f1_score[1:])
    mean_iou = np.mean(iou[1:]) 
    mcc = compute_mcc(cm[1:,1:])
    pop = np.sum(cm)

    with open(os.path.join(DIR_TILES_PRED,FILE_NAME_ADDENDUM+'metrics.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        title= ['class_id',	'class_name', 'iou', 'mcc',	'f1', 'precision','recall', 'pop', 'pop%']
        row_mean = ['', 'mean', f"{mean_iou:.2f}", f"{mcc:.2f}", f"{macro_f1:.2f}", f"{macro_precision:.2f}", f"{macro_recall:.2f}", f"{pop}", '100']
        writer.writerows([title, row_mean])
        for id in np.arange(0, n_classes):
            row = [id, DICT_CLASSES[id], f"{iou[id]:.2f}", '', f"{f1_score[id]:.2f}", f"{precision[id]:.2f}", f"{recall[id]:.2f}", f"{labels_per_class[id]}", f"{labels_per_class[id]/pop*100:.1f}"]
            writer.writerows([row])
            logger.info(f"IoU {DICT_CLASSES[id]}={iou[id]:.2f}")

    logger.info(f"Accuracy: {accuracy:.2f}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1-score: {f1_score}")
    logger.info(f"IoU: {mean_iou:.2f}")
    logger.info(f"Macro Precision: {macro_precision:.2f}")
    logger.info(f"Macro Recall: {macro_recall:.2f}")
    logger.info(f"Macro F1-score: {macro_f1:.2f}")
    logger.info(f"Mean IoU: {mean_iou:.2f}")
    logger.info(f"MCC: {mcc:.2f}")

    return

if __name__ == "__main__":

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script computes confusion matrix per tiles and derives metrics for the whole set...")
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
    FILE_NAME_PREFIX = cfg["file_name_prefix"]
    DICT_CLASSES = cfg["dict_classes"]
    AREA_TYPE_FILE = cfg["area_type_file"] 
    AREA_TYPE = cfg["area_type"] 
    LOG_FILE = cfg["log_file"] 

    # set up logger
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(LOG_FILE, level="INFO")
    
    logger.info("---------------------")
    logger.info(f"{DIR_TILES_LABEL = }")
    logger.info(f"{DIR_TILES_PRED = }")    
    logger.info(f"{DICT_CLASSES = }")    
    logger.info(f"{AREA_TYPE_FILE}")

    FILE_NAME_ADDENDUM = FILE_NAME_PREFIX+'_'+str(len(DICT_CLASSES))+'_'+str(AREA_TYPE)+'_'
    metrics_from_cm(DIR_TILES_LABEL, DIR_TILES_PRED, FILE_NAME_ADDENDUM, DICT_CLASSES, AREA_TYPE, AREA_TYPE_FILE)
