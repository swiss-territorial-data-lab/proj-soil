import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import os
import sys
import yaml
import argparse
from loguru import logger
# import warnings

plt.rcParams["text.usetex"] = True
np.set_printoptions(linewidth=500, precision=3, suppress=True)


def create_or_append(dic, key, value):
    if key not in dic:
        dic[key] = [value]
        return dic
    dic[key].append(value)
    return dic


def create_or_extend(dic, key, value):
    if key not in dic:
        dic[key] = value
        return dic
    dic[key].extend(value)
    return dic


def calculate_metrics(
    PRED_FOLDER,
    GT_FOLDER,
    CONF_MATRIX_MODEL,
    CLASSES,
    SOIL_CLASSES,
    CREATE_CM,
    SAME_NAMES,
    METRIC_CSV_PATH_MULTICLASS,
    METRIC_CSV_PATH_BINARY,
    COUNT_CSV_PATH_MULTICLASS,
    COUNT_CSV_PATH_BINARY,
    CONF_MATRIX_PATH_MULTICLASS,
    CONF_MATRIX_PATH_BINARY,
        EXCLUDE_IDS) -> None:

    """
    Calculate evaluation metrics and confusion matrices.

    This function calculates the IoU metrics and confusion matrices for
    predicted and ground truth label images. The results are logged and
    visualized.

    Parameters:
    - PRED_FOLDER (str): Path to folder containing predicted label images.
    - GT_FOLDER (str): Path to folder containing ground truth label images.
    - CONF_MATRIX_MODEL (str): Model name for which the confusion matrix is
    generated.
    - CLASSES (list): List of class names.
    - SOIL_CLASSES (list): List of soil class names.
    - CREATE_CM (bool): Flag indicating whether to create confusion matrix
    visualization.
    - SAME_NAMES (bool): Flag indicating whether predicted and ground truth
    images have the same filenames.
    - METRIC_CSV_PATH_MULTICLASS (str): Path to save the metric results for
    multiclass classification as a CSV file.
    - METRIC_CSV_PATH_BINARY (str): Path to save the metric results for binary
    classification as a CSV file.
    - COUNT_CSV_PATH_MULTICLASS (str): Path to save the count results for
    multiclass classification as a CSV file.
    - COUNT_CSV_PATH_BINARY (str): Path to save the count results for binary
    classification as a CSV file.
    - CONF_MATRIX_PATH_MULTICLASS (str): Path to save the confusion matrix
    visualization for multiclass classification.
    - CONF_MATRIX_PATH_BINARY (str): Path to save the confusion matrix
    visualization for binary classification.
    - EXCLUDE_IDS (list): List of IDs to exclude from the calculation.

    Returns:
    - tuple: A tuple containing the metric results and count results.
    """

    arch_tps = {"multiclass": {}, "binary": {}}
    arch_fps = {"multiclass": {}, "binary": {}}
    arch_fns = {"multiclass": {}, "binary": {}}
    arch_tns = {"multiclass": {}, "binary": {}}

    # store flattened arrays to use for the confusion matrix
    flat_gts = {"multiclass": [], "binary": []}
    flat_preds = {"multiclass": {}, "binary": {}}

    # Iterate through ground truth label images
    for gt_filename in os.listdir(GT_FOLDER):
        if not gt_filename.endswith((".tif", ".tiff")):
            continue

        # Extract the unique identifier from the very end of the filename
        id = gt_filename.split(".")[0].split("_")[-1]

        EXCLUDE_IDS = [str(id) for id in EXCLUDE_IDS]
        if id in EXCLUDE_IDS:
            continue

        # a switch to store the flattened groundtruth only if there was a matching prediction
        flat_switch = 1

        # Iterate through predicted label images to find matching prediction
        for root, dirs, files in os.walk(PRED_FOLDER):
            if SAME_NAMES:
                matching_pred = [
                    prediction for prediction in files if prediction == gt_filename
                ]
            else:
                matching_pred = [
                    prediction
                    for prediction in files
                    if prediction.endswith((f"_{id}.tif", f"_{id}.tiff"))
                ]

            if len(matching_pred) == 0:
                continue

            # there should only be exactly one matching prediction
            assert len(matching_pred) == 1

            if flat_switch == 1:
                # Read the ground truth label image
                with rasterio.open(os.path.join(GT_FOLDER, gt_filename)) as gt_file:
                    groundtruth = gt_file.read(1)
                    groundtruth_binary = np.isin(groundtruth, SOIL_CLASSES)

                flat_gts["multiclass"].extend(list(groundtruth.flatten()))
                flat_gts["binary"].extend(list(groundtruth_binary.flatten()))
                flat_switch = 0

            # Load and read the predicted label image
            pred_filename = matching_pred[0]
            with rasterio.open(os.path.join(root, pred_filename)) as pred_file:
                prediction = pred_file.read(1, masked=True)
                mask = prediction.mask

                prediction_binary = np.isin(prediction, SOIL_CLASSES)
                prediction_binary = np.ma.masked_array(prediction_binary, mask=mask)

            class_tps = {"multiclass": [], "binary": []}
            class_fps = {"multiclass": [], "binary": []}
            class_fns = {"multiclass": [], "binary": []}
            class_tns = {"multiclass": [], "binary": []}

            for mode, num_classes, pred, gt in zip(
                ["multiclass", "binary"],
                [len(CLASSES), 2],
                [prediction, prediction_binary],
                [groundtruth, groundtruth_binary],
            ):

                for c in range(num_classes):
                    if mode == "multiclass":
                        c += 1
                    class_tps[mode].append(np.sum((pred == c) & (gt == c)))
                    class_fps[mode].append(np.sum((pred == c) & (gt != c)))
                    class_fns[mode].append(np.sum((pred != c) & (gt == c)))
                    class_tns[mode].append(np.sum((pred != c) & (gt != c)))

                # Determine the architecture of the prediction for organizing
                # results by looking at the root folder
                arch = root.split("/")[-1]

                # create_or_append(dic, key, value)
                arch_tps[mode] = create_or_append(arch_tps[mode], arch, class_tps[mode])
                arch_fps[mode] = create_or_append(arch_fps[mode], arch, class_fps[mode])
                arch_fns[mode] = create_or_append(arch_fns[mode], arch, class_fns[mode])
                arch_tns[mode] = create_or_append(arch_tns[mode], arch, class_tns[mode])

            # create_or_extend(dic, key, value)
            flat_preds["multiclass"] = create_or_extend(
                flat_preds["multiclass"], arch, list(prediction.flatten())
            )
            flat_preds["binary"] = create_or_extend(
                flat_preds["binary"], arch, list(prediction_binary.flatten())
            )

    archs = list(arch_tps["multiclass"].keys())
    archs.sort()

    logger.info(
        f"{len(arch_tps['multiclass'][archs[0]])} images with shape {prediction.shape} have been considered.\n"
    )

    csv = {
        "multiclass": "type,arch,class_id,class_name,iou,mcc,f1,precision,recall\n",
        "binary": "type,arch,class_id,class_name,iou,mcc,f1,precision,recall\n",
    }

    # Iterate through architectures and calculate mIoU and related metrics
    for arch in archs:

        for mode in ["multiclass", "binary"]:

            # multiclass metrics
            class_tps = np.array(arch_tps[mode][arch], dtype=float).sum(axis=0)
            class_fps = np.array(arch_fps[mode][arch], dtype=float).sum(axis=0)
            class_fns = np.array(arch_fns[mode][arch], dtype=float).sum(axis=0)
            class_tns = np.array(arch_tns[mode][arch], dtype=float).sum(axis=0)
            class_unions = class_tps + class_fps + class_fns

            # class_tps[class_unions == 0] = np.nan
            # class_fps[class_unions == 0] = np.nan
            # class_fns[class_unions == 0] = np.nan
            # class_tns[class_unions == 0] = np.nan
            # class_unions[class_unions == 0] = np.nan

            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore", category=RuntimeWarning)
            class_precisions = class_tps / (class_tps + class_fps)
            class_recalls = class_tps / (class_tps + class_fns)
            class_f1s = class_tps / (class_tps + (0.5 * (class_fps + class_fns)))
            class_ious = class_tps / (class_unions)
            class_mccs = (class_tps * class_tns - class_fps * class_fns) / np.sqrt(
                (class_tps + class_fps)
                * (class_tps + class_fns)
                * (class_tns + class_fps)
                * (class_tns + class_fns)
            )

            mean_precision = np.round(np.nanmean(class_precisions), 3)
            mean_recall = np.round(np.nanmean(class_recalls), 3)
            mean_f1 = np.round(np.nanmean(class_f1s), 3)
            mean_iou = np.round(np.nanmean(class_ious), 3)
            mean_mcc = np.round(np.nanmean(class_mccs), 3)

            csv[
                mode
            ] += f"{mode},{arch},None,mean,{mean_iou},{mean_mcc},{mean_f1},{mean_precision},{mean_recall}\n"

            logger.info(f"** {arch} **:")

            if mode == "multiclass":
                classes = CLASSES
            elif mode == "binary":
                classes = ["Non-soil", "Soil"]

            # logger.info(f"Class Metrics:")
            # with warnings.catch_warnings():
            # warnings.simplefilter("ignore", category=RuntimeWarning)
            for i, clas in enumerate(classes):
                precision = class_precisions.round(3)[i]
                recall = class_recalls.round(3)[i]
                f1 = class_f1s.round(3)[i]
                iou = class_ious.round(3)[i]
                mcc = class_mccs.round(3)[i]

                # logger.info(f"{f'{clas} ({i + 1})':<21}: " +
                #     f"IoU = {mc_iou:<5}, F1 = {mc_f1:<5}, Precision = {mc_precision:<5}, Recall = {mc_recall}")

                csv[
                    mode
                ] += f"{mode},{arch},{i+1},{clas},{iou},{mcc},{f1},{precision},{recall}\n"

            logger.info("")
            logger.info(f"Mean {mode} Metrics:")
            logger.info(f"{mean_iou = }")
            logger.info(f"{mean_mcc = }")
            logger.info(f"{mean_f1 = }")
            logger.info(f"{mean_precision = }")
            logger.info(f"{mean_recall = }\n")

    with open(METRIC_CSV_PATH_MULTICLASS, "w") as file:
        file.write(csv["multiclass"])
    with open(METRIC_CSV_PATH_BINARY, "w") as file:
        file.write(csv["binary"])

    for mode in ["multiclass", "binary"]:

        groundtruth = flat_gts[mode]
        prediction = flat_preds[mode][CONF_MATRIX_MODEL]
        if mode == "multiclass":

            count_csv_path = COUNT_CSV_PATH_MULTICLASS
            count_csv = "data,type," + ",".join(CLASSES) + "\n"

            # # add one value from each class to the flattened images to ensure
            # # at least one occurence of all classes

            with_zero = False
            if 0 in np.unique(prediction):
                with_zero = True

            first_number = 0 if with_zero else 1
            groundtruth = np.array(
                groundtruth + list(range(first_number, len(CLASSES) + 1))
            )
            prediction = np.array(
                prediction + list(range(first_number, len(CLASSES) + 1))
            )

            prediction = prediction[groundtruth != 0]
            groundtruth = groundtruth[groundtruth != 0]

        elif mode == "binary":
            count_csv_path = COUNT_CSV_PATH_BINARY
            count_csv = "data,type,non-soil,soil\n"

            groundtruth = np.array(groundtruth + [0, 1])
            prediction = np.array(prediction + [0, 1])

            flat_preds_array = np.array(prediction).astype(np.float32)
            prediction = flat_preds_array[~np.isnan(flat_preds_array)]
            groundtruth = np.array(groundtruth).astype(float)[
                ~np.isnan(flat_preds_array)
            ]

        _, gt_counts = np.unique(groundtruth, return_counts=True)
        _, pred_counts = np.unique(prediction, return_counts=True)
        count_csv += (
            f"groundtruth,{mode}," + ",".join([str(el) for el in gt_counts]) + "\n"
        )
        count_csv += (
            f"prediction,{mode}," + ",".join([str(el) for el in pred_counts]) + "\n"
        )

        with open(count_csv_path, "w") as file:
            file.write(count_csv)

        if CREATE_CM:
            from sklearn.metrics import (
                confusion_matrix,
            )  # CM: is that a common good practice? Or better to test/load all the packages at the beginning?
            import pandas as pd

            # Compute the confusion matrix
            cm = confusion_matrix(groundtruth, prediction)

            # Calculate the percentage for each cell
            # cm_percentage_total = cm.astype('float') / cm.sum()
            cm_percentage_actual = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            cm_percentage_pred = cm.astype("float") / cm.sum(axis=0)[np.newaxis, :]

            # Create a DataFrame for the confusion matrix
            if mode == "multiclass":
                labels = ["unknown"] + CLASSES if with_zero else CLASSES
                cm_df = pd.DataFrame(cm_percentage_actual, index=labels, columns=labels)

                # Plotting using Seaborn's heatmap

                plt.figure(figsize=(18, 9))

                # Create the heatmap
                ax = sns.heatmap(
                    cm_df, annot=False, fmt=".2%", cbar=False, cmap="Blues"
                )

                # Overlay colored patches for TP, TN, FP, and FN
                for i in range(cm_df.shape[0]):
                    for j in range(cm_df.shape[1]):

                        # Add text annotation
                        text = (
                            rf"\begin{{tabular}}{{l}}"  # Begin tabular environment, aligned left
                            # rf"\textbf{{{textlabel}}}\\\\"  # Bold line 1
                            rf"\textbf{{{cm[i, j]:,} Pixels}}\\"  # Bold line 2
                            rf"\textbf{{Rel. to GT: {cm_percentage_actual[i, j]*100:.2f}\%}}\\"  # Bold line 3
                            # rf"Relative to GT: {cm_percentage_actual[i, j]*100:.2f}\%"  # Normal line 1
                            rf"Rel. p.c. Pred: {cm_percentage_pred[i, j]*100:.2f}\%"  # Normal line 2
                            rf"\end{{tabular}}"  # End tabular environment
                        )

                        if cm[i, j] in [0, 1]:
                            ax.add_patch(
                                plt.Rectangle(
                                    (j, i), 1, 1, fill=True, color="white", lw=0
                                )
                            )
                        else:
                            textcolor = (
                                "black"
                                if cm_percentage_actual[i, j] <= 0.5
                                else "white"
                            )
                            ax.text(
                                j + 0.5,
                                i + 0.5,
                                text,
                                ha="center",
                                va="center",
                                color=textcolor,
                                fontsize="small",
                            )

                # Draw the grid lines after all other plotting commands
                for i in range(cm.shape[0] + 1):
                    ax.hlines(
                        i,
                        xmin=0,
                        xmax=cm.shape[1],
                        colors="black",
                        linestyles="solid",
                        linewidths=0.5,
                    )
                for j in range(cm.shape[1] + 1):
                    ax.vlines(
                        j,
                        ymin=0,
                        ymax=cm.shape[0],
                        colors="black",
                        linestyles="solid",
                        linewidths=0.5,
                    )

                # ATTENTION: Classes for soil and non-soil are hard-coded here, starting with 6, anding with 6+6=12
                ax.add_patch(
                    plt.Rectangle(
                        (6.01, 0.03), 5.98, 5.94, fill=False, edgecolor="red", linewidth=2
                    )
                )
                ax.add_patch(
                    plt.Rectangle(
                        (0.01, 6.03), 5.98, 5.94, fill=False, edgecolor="red", linewidth=2
                    )
                )
                # Adjust plot boundaries if needed
                ax.set_xlim(0, cm.shape[1] + 0.01)  # Adjusting x-axis limit
                ax.set_ylim(0, cm.shape[0] + 0.01)  # Adjusting y-axis limit

                ax.invert_yaxis()

                # Find the colors within the heatmap where
                # Labels, title and ticks
                # Move x-axis ticks and labels to the top
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position("top")
                num_classes = len(CLASSES) + 1 if with_zero else len(CLASSES)
                plt.xticks(np.arange(num_classes) + 0.5, labels, rotation=30)
                plt.yticks(np.arange(num_classes) + 0.5, labels, rotation=30)
                plt.title(
                    rf'\textbf{{Confusion Matrix of the model "{CONF_MATRIX_MODEL}" for Multiclass Classification}}'
                    + "\nColoured relative to GT, The red rectangles mark soil-nonsoil-confusions"
                )
                plt.ylabel(r"\textbf{Ground Truth}")
                plt.xlabel("\n" + r"\textbf{Predicted}")

                # Show the plot
                plt.tight_layout()
                # plt.show()
                plt.savefig(CONF_MATRIX_PATH_MULTICLASS)
                plt.close()

            elif mode == "binary":
                labels = ["Non-soil", "Soil"]
                cm_df = pd.DataFrame(cm_percentage_actual, index=labels, columns=labels)

                # Plotting using Seaborn's heatmap
                plt.figure(figsize=(10, 8))

                # Create the heatmap
                ax = sns.heatmap(
                    cm_df, annot=False, fmt=".2%", cbar=False, cmap="Blues"
                )

                # Overlay colored patches for TP, TN, FP, and FN
                for i in range(cm_df.shape[0]):
                    for j in range(cm_df.shape[1]):
                        if i == j:  # True positives and true negatives
                            color = "#C1F0B2"
                            textlabel = "TN" if i == 0 else "TP"
                            textcolor = "white"
                        else:  # False positives and false negatives
                            color = "#FF9AA2"
                            textlabel = "FN" if i > j else "FP"
                            textcolor = "black"
                        rect = Rectangle(
                            (j + 0.01, i + 0.01),
                            0.98,
                            0.98,
                            fill=False,
                            color=color,
                            alpha=1,
                            linewidth=7,
                        )
                        ax.add_patch(rect)

                        # Add text annotation
                        text = (
                            rf"\begin{{tabular}}{{l}}"  # Begin tabular environment, aligned left
                            rf"\textbf{{{textlabel}}}\\\\"  # Bold line 1
                            rf"\textbf{{{cm[i, j]:,} Pixels}}\\"  # Bold line 2
                            rf"\textbf{{Relative to per-class GT: {cm_percentage_actual[i, j]*100:.2f}\%}}\\"  # Bold line 3
                            rf"Relative to per-class Pred: {cm_percentage_pred[i, j]*100:.2f}\%"  # Normal line 2
                            rf"\end{{tabular}}"
                        )  # End tabular environment
                        ax.text(
                            j + 0.5,
                            i + 0.5,
                            text,
                            ha="center",
                            va="center",
                            color=textcolor,
                            fontsize="large",
                        )

                # Labels, title and ticks
                # Move x-axis ticks and labels to the top
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position("top")
                label_names = ["Non-Soil", "Soil"]
                plt.xticks(np.arange(len(label_names)) + 0.5, label_names, rotation=30)
                plt.yticks(np.arange(len(label_names)) + 0.5, label_names, rotation=30)
                plt.title(
                    rf'\textbf{{Confusion Matrix of Model "{CONF_MATRIX_MODEL}" for Binary Classification}} (Coloured relative to GT)'
                )
                plt.ylabel(r"\textbf{Ground Truth}")
                plt.xlabel("\n" + r"\textbf{Prediction}")

                # Show the plot
                plt.tight_layout()
                # plt.show()
                plt.savefig(CONF_MATRIX_PATH_BINARY)


if __name__ == "__main__":

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script...")
    parser.add_argument(
        "-cfg",
        "--config_file",
        type=str,
        help="Framework configuration file",
        default="/Users/nicibe/Desktop/Job/swisstopo_stdl/soil_fribourg/proj-soils/config/eval/config-eval_heigvd-32k.yaml",
    )
    # default="/Users/nicibe/Desktop/Job/swisstopo_stdl/soil_fribourg/proj-soils/config/config-eval_heig-vd_finetuned.yaml")
    args = parser.parse_args()

    # load input parameters
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    PRED_FOLDER = cfg["pred_folder"]
    GT_FOLDER = cfg["gt_folder"]
    CONF_MATRIX_MODEL = cfg["conf_matrix_model"]
    CLASSES = cfg["classes"]
    SOIL_CLASSES = cfg["soil_classes"]
    CREATE_CM = cfg["create_cm"]
    SAME_NAMES = cfg["same_names"]
    METRIC_CSV_PATH_MULTICLASS = cfg["metric_csv_path_multiclass"]
    COUNT_CSV_PATH_MULTICLASS = cfg["count_csv_path_multiclass"]
    CONF_MATRIX_PATH_MULTICLASS = cfg["conf_matrix_path_multiclass"]
    METRIC_CSV_PATH_BINARY = cfg["metric_csv_path_binary"]
    COUNT_CSV_PATH_BINARY = cfg["count_csv_path_binary"]
    CONF_MATRIX_PATH_BINARY = cfg["conf_matrix_path_binary"]
    EXCLUDE_IDS = cfg["exclude_ids"]
    LOG_FILE = cfg["log_file"]

    # set up logger
    logger_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> |\
                    <level>{level: <8}</level> | <cyan>{function}</cyan>\
                    :<cyan>{line}</cyan> - <level>{message}</level>"
    logger.remove()
    logger.add(sys.stdout, level="INFO", format=logger_format)
    logger.add(LOG_FILE, level="INFO", format=logger_format)

    logger.info(f"{PRED_FOLDER = }")
    logger.info(f"{GT_FOLDER = }")
    logger.info(f"{CONF_MATRIX_MODEL = }")
    logger.info(f"{CLASSES = }")
    logger.info(f"{SOIL_CLASSES = }")
    logger.info(f"{CREATE_CM = }")
    logger.info(f"{SAME_NAMES = }")
    logger.info(f"{METRIC_CSV_PATH_MULTICLASS = }")
    logger.info(f"{METRIC_CSV_PATH_BINARY = }")
    logger.info(f"{COUNT_CSV_PATH_MULTICLASS = }")
    logger.info(f"{COUNT_CSV_PATH_BINARY = }")
    logger.info(f"{CONF_MATRIX_PATH_MULTICLASS = }")
    logger.info(f"{CONF_MATRIX_PATH_BINARY = }")
    logger.info(f"{EXCLUDE_IDS = }")
    logger.info(f"{LOG_FILE = }")

    # run program
    logger.info("Started Programm")

    calculate_metrics(
        PRED_FOLDER,
        GT_FOLDER,
        CONF_MATRIX_MODEL,
        CLASSES,
        SOIL_CLASSES,
        CREATE_CM,
        SAME_NAMES,
        METRIC_CSV_PATH_MULTICLASS,
        METRIC_CSV_PATH_BINARY,
        COUNT_CSV_PATH_MULTICLASS,
        COUNT_CSV_PATH_BINARY,
        CONF_MATRIX_PATH_MULTICLASS,
        CONF_MATRIX_PATH_BINARY,
        EXCLUDE_IDS,
    )
    logger.info("Ended Program\n")
