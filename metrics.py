import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import wilcoxon
import gudhi
import torch
# Dice Similarity Coefficient (DSC)
def dice_coefficient(y_pred, y_true, default_value=0.0):
    """
    Calculate Dice coefficient between two binary masks.
    If both masks are empty, returns 1.0 (perfect match).
    If only one mask is empty, returns default_value.
    
    Args:
        y_pred: Binary prediction mask
        y_true: Binary ground truth mask
        default_value: Value to return when only one mask is empty (default: 0.0)
    """
    pred_sum = np.sum(y_pred)
    true_sum = np.sum(y_true)
    
    # Both masks are empty - perfect match
    if pred_sum == 0 and true_sum == 0:
        return 1.0
    # Only one mask is empty - return default value
    if pred_sum == 0 or true_sum == 0:
        return default_value
        
    intersection = np.sum(y_pred * y_true)
    return 2.0 * intersection / (pred_sum + true_sum)

def generalized_dice(pred, target, num_classes=4, epsilon=1e-6):
    """
    Compute the Generalized Dice Similarity Coefficient (gDSC) for multi-class segmentation.

    Args:
        pred (torch.Tensor): Predicted segmentation (B, C, H, W) as softmax output.
        target (torch.Tensor): Ground truth segmentation (B, H, W) as class index map.
        num_classes (int): Number of segmentation classes.
        epsilon (float): Small value to prevent division by zero.

    Returns:
        gDSC (float): Generalized Dice Score
        class_gDSC (list): Per-class Dice Scores
    """
    # Ensure target is one-hot encoded
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

    # Flatten along spatial dimensions
    pred_flat = pred.view(pred.shape[0], num_classes, -1)  # (B, C, H*W)
    target_flat = target_one_hot.view(target_one_hot.shape[0], num_classes, -1)

    # Compute class weights (Inverse squared frequency)
    class_weights = 1.0 / (torch.sum(target_flat, dim=2) ** 2 + epsilon)

    # Compute intersection
    intersection = 2 * torch.sum(target_flat * pred_flat, dim=2)

    # Compute denominator
    denominator = torch.sum(target_flat, dim=2) + torch.sum(pred_flat, dim=2) + epsilon

    # Compute per-class DSC
    class_dice = intersection / denominator  # Shape: (B, C)

    # Compute weighted sum for generalized DSC
    gDSC = torch.sum(class_weights * intersection, dim=1) / torch.sum(class_weights * denominator, dim=1)

    return gDSC.mean().item(), class_dice.mean(dim=0).tolist()
# Hausdorff Distance (HDD)
def hausdorff_distance(y_pred, y_true, default_value=float('inf')):
    """
    Calculate the Hausdorff distance between two binary masks.
    If either mask is empty, returns default_value.
    
    Args:
        y_pred: Binary prediction mask
        y_true: Binary ground truth mask
        default_value: Value to return when either mask is empty (default: inf)
    """
    pred_points = np.argwhere(y_pred > 0)
    true_points = np.argwhere(y_true > 0)
    
    if len(pred_points) == 0 or len(true_points) == 0:
        return default_value
        
    return max(
        directed_hausdorff(pred_points, true_points)[0],
        directed_hausdorff(true_points, pred_points)[0]
    )

# Betti Error (BE) - Requires precomputed Betti numbers
# Here, we simulate Betti numbers as an example
# Replace `betti_pred` and `betti_true` with actual Betti numbers

def betti_error(betti_pred, betti_true):
    """
    Compute Betti Error (BE) between predicted and ground truth Betti numbers
    following Equation (35) in the paper.

    Args:
        betti_pred (dict): Predicted Betti numbers for each class and combination.
        betti_true (dict): Ground truth Betti numbers for each class and combination.

    Returns:
        int: Betti error (BE)
    """
    betti_error_total = 0
    for key in betti_true.keys():  # Ensure both single-class & combinations are checked
        betti_error_total += np.sum(np.abs(np.array(betti_pred[key]) - np.array(betti_true[key])))
    return betti_error_total



# Topological Success (TS)
def topological_success(betti_error):
    return 1 if betti_error == 0 else 0

# Statistical Tests
# Wilcoxon Signed-Rank Test
def wilcoxon_signed_rank_test(metric1, metric2):
    return wilcoxon(metric1, metric2)

def compute_betti_numbers(binary_image, max_dim=1):
    """
    Compute Betti numbers for a binary image using persistent homology.
    
    Args:
        binary_image (np.ndarray): Binary image (0s and 1s)
        max_dim (int): Maximum homology dimension to compute (default=2)
    
    Returns:
        list: Betti numbers [β₀, β₁]
    """
    # Ensure binary image is properly formatted
    binary_image = binary_image.astype(np.float64)
    
    # Create a cubical complex from the binary image
    # GUDHI considers 0s as foreground, so we invert the image
    cubical_complex = gudhi.CubicalComplex(
        top_dimensional_cells=1 - binary_image
    )
    
    # Compute persistence with homology over Z2 field
    persistence = cubical_complex.persistence(homology_coeff_field=2)
    
    # Initialize Betti numbers
    betti = [0] * (max_dim + 1)
    
    # Count features that persist
    if persistence is not None:
        for dim, (birth, death) in persistence:
            if dim <= max_dim:
                if death == float('inf') or (death - birth) > 0.5:  # Only count significant features
                    betti[dim] += 1
    
    return betti

def compute_class_combinations_betti(segmentation, max_dim=2):
    """
    Compute Betti numbers for individual classes and their pair combinations.
    
    Args:
        segmentation (np.ndarray): Segmentation mask with class labels
        max_dim (int): Maximum homology dimension to compute (default=2)
    
    Returns:
        dict: Dictionary containing Betti numbers for classes and combinations
    """
    combinations = {
        (1,): None,   # RV
        (2,): None,   # MYO
        (3,): None,   # LV
        (1, 2): None, # RV + MYO
        (1, 3): None, # RV + LV
        (2, 3): None  # MYO + LV
    }
    
    # Compute Betti numbers for single classes
    for class_idx in range(1, 4):
        # Create binary mask and remove small components
        mask = (segmentation == class_idx)
        if np.sum(mask) > 0:  # Only compute if class exists in the image
            combinations[(class_idx,)] = compute_betti_numbers(mask, max_dim)
        else:
            combinations[(class_idx,)] = [0] * (max_dim + 1)
    
    # Compute Betti numbers for class pairs
    pairs = [(1, 2), (1, 3), (2, 3)]
    for pair in pairs:
        mask = np.logical_or(
            segmentation == pair[0],
            segmentation == pair[1]
        )
        if np.sum(mask) > 0:  # Only compute if combination exists
            combinations[pair] = compute_betti_numbers(mask, max_dim)
        else:
            combinations[pair] = [0] * (max_dim + 1)
    
    return combinations

def compute_percentiles(metric_list, percentiles=[25, 50, 75, 98, 99, 100]):
    return np.percentile(metric_list, percentiles)

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
    gdsc_mean, gdsc_per_class = generalized_dice(y_pred, y_true)

    # Calculate HDD
    hdd_per_class = [hausdorff_distance(y_pred[i], y_true[i]) for i in range(len(y_pred))]

    # Example Betti numbers
    betti_pred = [1, 1, 0]  # Example predicted Betti numbers
    betti_true = [1, 1, 0]  # Example ground truth Betti numbers
    be = betti_error(betti_pred, betti_true)
    ts = topological_success(be)

    print("Per-Class DSC:", dsc_per_class)
    print("Generalized DSC (mean):", gdsc_mean)
    print("Generalized DSC (per class):", gdsc_per_class)
    print("Per-Class HDD:", hdd_per_class)
    print("Betti Error:", be)
    print("Topological Success:", ts)

if __name__ == "__main__":
    example_metrics()
