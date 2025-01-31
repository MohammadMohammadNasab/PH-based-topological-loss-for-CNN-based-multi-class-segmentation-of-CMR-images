from skimage.measure import label
import numpy as np

def connected_component_analysis(pred_mask, connectivity):
    """
    Keep only the largest connected component in the predicted mask.
    
    Args:
        pred_mask (np.ndarray): Binary predicted mask of shape (H, W)
    
    Returns:
        np.ndarray: Mask with only the largest connected component
    """
    # Label connected components
    labeled_mask = label(pred_mask, connectivity)

    # If no components detected, return original mask
    if labeled_mask.max() == 0:
        return np.zeros_like(pred_mask, dtype=np.uint8)

    # Compute component sizes (ignore background, which is label 0)
    component_sizes = np.bincount(labeled_mask.ravel(), minlength=labeled_mask.max() + 1)[1:]

    # Get the largest connected component label
    largest_label = np.argmax(component_sizes) + 1

    # Create a mask with only the largest component
    largest_component = (labeled_mask == largest_label).astype(np.uint8)

    return largest_component
