import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import wilcoxon
import gudhi

# Dice Similarity Coefficient (DSC)
def dice_coefficient(y_pred, y_true):
    intersection = np.sum(y_pred * y_true)
    return 2.0 * intersection / (np.sum(y_pred) + np.sum(y_true))

# Generalized Dice Similarity Coefficient (gDSC)
def generalized_dice(y_preds, y_trues, epsilon=1e-7):
    """
    Calculate generalized dice score.
    
    Args:
        y_preds: predictions of shape (B, C, H, W) - softmax probabilities
        y_trues: ground truth of shape (B, H, W) with class indices
        epsilon: small constant to avoid division by zero
    
    Returns:
        tuple (mean_dice, class_dice) where class_dice is array of dice scores per class
    """
    n_classes = y_preds.shape[1]
    
    # Convert y_trues to one-hot encoding
    y_trues_one_hot = np.zeros_like(y_preds)
    for i in range(n_classes):
        y_trues_one_hot[:, i, ...] = (y_trues == i)
    
    # Calculate intersection and union
    intersection = np.sum(y_preds * y_trues_one_hot, axis=(0, 2, 3))
    union = np.sum(y_preds, axis=(0, 2, 3)) + np.sum(y_trues_one_hot, axis=(0, 2, 3))
    
    # Calculate dice score for each class
    class_dice = (2.0 * intersection + epsilon) / (union + epsilon)
    mean_dice = np.mean(class_dice)
    
    return mean_dice, class_dice

# Hausdorff Distance (HDD)
def hausdorff_distance(y_pred, y_true):
    """
    Calculate the Hausdorff distance between two binary masks.
    
    Args:
        y_pred: Binary prediction mask
        y_true: Binary ground truth mask
    
    Returns:
        float: Hausdorff distance
    """
    pred_points = np.argwhere(y_pred > 0)
    true_points = np.argwhere(y_true > 0)
    
    if len(pred_points) == 0 or len(true_points) == 0:
        return float('inf')
        
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

def compute_betti_numbers(binary_image, max_dim=2):
    """
    Compute Betti numbers for a binary image using persistent homology.
    
    Args:
        binary_image (np.ndarray): Binary image (0s and 1s)
        max_dim (int): Maximum homology dimension to compute (default=2)
    
    Returns:
        list: Betti numbers [β₀, β₁, β₂]
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
