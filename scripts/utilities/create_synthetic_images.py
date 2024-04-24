import os
import cv2

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def create_images(safe_dir=None):
    # Create a base 512x512 image filled with zeros (background class)
    gt = np.zeros((512, 512), dtype=np.uint8)
    pred = np.zeros((512, 512), dtype=np.uint8)

    # Create square patches for class 1 (Object 1) in ground truth and prediction
    gt[100:300, 100:300] = 1
    pred[150:350, 150:350] = 1

    # Create square patches for class 2 (Object 2) in ground truth and prediction
    gt[300:500, 300:500] = 1
    pred[320:520, 320:520] = 1

    num_classes = 2
    total_iou = 0
    total_precision = 0
    total_recall = 0

    for c in range(num_classes):
        intersection = np.sum((gt == c) & (pred == c))
        union = np.sum((gt == c) | (pred == c))
        iou = intersection / union

        tp = intersection
        fp = np.sum((gt != c) & (pred == c))
        fn = np.sum((gt == c) & (pred != c))
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0

        total_iou += iou
        total_precision += precision
        total_recall += recall

        print(f"Class {c} - IoU: {iou:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")

    mean_iou = total_iou / num_classes
    mean_precision = total_precision / num_classes
    mean_recall = total_recall / num_classes
    accuracy = np.sum(gt == pred) / (512 * 512)

    print(f"\nMean IoU: {mean_iou:.2f}")
    print(f"Mean Precision: {mean_precision:.2f}")
    print(f"Mean Recall: {mean_recall:.2f}")
    print(f"Accuracy: {accuracy:.2f}")

    # Display the images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(gt, cmap='tab10', vmin=0, vmax=2)
    axes[0].set_title("Ground Truth")
    axes[1].imshow(pred, cmap='tab10', vmin=0, vmax=2)
    axes[1].set_title("Prediction")
    plt.tight_layout()
    plt.show()


    if safe_dir is not None:
        cv2.imwrite(os.path.join(safe_dir, "gt.tif"), gt)
        cv2.imwrite(os.path.join(safe_dir, "pred.tif"), pred)

    print(np.mean(pred))

# create_images(".")
create_images()