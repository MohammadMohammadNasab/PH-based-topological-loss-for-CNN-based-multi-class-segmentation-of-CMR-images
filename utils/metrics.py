import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import wilcoxon

# Dice Similarity Coefficient (DSC)
def dice_coefficient(y_pred, y_true):
    intersection = np.sum(y_pred * y_true)
    return 2.0 * intersection / (np.sum(y_pred) + np.sum(y_true))

# Generalized Dice Similarity Coefficient (gDSC)
def generalized_dice(y_preds, y_trues, epsilon=1e-6):
    if y_trues.ndim == 3:
        num_classes = y_preds.shape[1]
        y_trues = np.eye(num_classes)[y_trues]  # One-hot encode
        y_trues = y_trues.transpose(0, 3, 1, 2)
    
    intersection = (y_preds * y_trues).sum(axis=(0, 2, 3))
    sums = y_preds.sum(axis=(0, 2, 3)) + y_trues.sum(axis=(0, 2, 3))
    dice = (2.0 * intersection + epsilon) / (sums + epsilon)
    return dice.mean()

# Hausdorff Distance (HDD)
def hausdorff_distance(y_pred, y_true):
    pred_points = np.argwhere(y_pred > 0)
    true_points = np.argwhere(y_true > 0)
    return max(
        directed_hausdorff(pred_points, true_points)[0],
        directed_hausdorff(true_points, pred_points)[0]
    )

# Betti Error (BE) - Requires precomputed Betti numbers
# Here, we simulate Betti numbers as an example
# Replace `betti_pred` and `betti_true` with actual Betti numbers

def betti_error(betti_pred, betti_true):
    return np.sum(np.abs(np.array(betti_pred) - np.array(betti_true)))

# Topological Success (TS)
def topological_success(betti_error):
    return 1 if betti_error == 0 else 0

# Statistical Tests
# Wilcoxon Signed-Rank Test
def wilcoxon_signed_rank_test(metric1, metric2):
    return wilcoxon(metric1, metric2)

# Example Usage
def example_metrics():
    # Example binary segmentations for 3 classes
    y_pred = [
        np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]]),
        np.array([[1, 1, 1], [0, 0, 0], [0, 1, 1]]),
        np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]])
    ]

    y_true = [
        np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]]),
        np.array([[1, 1, 0], [0, 0, 0], [0, 1, 1]]),
        np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]])
    ]

    # Calculate DSC and gDSC
    dsc_per_class = [dice_coefficient(y_pred[i], y_true[i]) for i in range(len(y_pred))]
    gdsc = generalized_dice(y_pred, y_true)

    # Calculate HDD
    hdd_per_class = [hausdorff_distance(y_pred[i], y_true[i]) for i in range(len(y_pred))]

    # Example Betti numbers
    betti_pred = [1, 1, 0]  # Example predicted Betti numbers
    betti_true = [1, 1, 0]  # Example ground truth Betti numbers
    be = betti_error(betti_pred, betti_true)
    ts = topological_success(be)

    print("Per-Class DSC:", dsc_per_class)
    print("Generalized DSC:", gdsc)
    print("Per-Class HDD:", hdd_per_class)
    print("Betti Error:", be)
    print("Topological Success:", ts)

if __name__ == "__main__":
    example_metrics()
