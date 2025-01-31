from skimage.measure import label
import numpy as np
def connected_component_analysis(pred_mask):
    """
    Keep only the largest connected component in the predicted mask.
    
    Args:
        pred_mask (np.ndarray): Predicted mask of shape (H, W)
    
    Returns:
        np.ndarray: Mask with only the largest connected component
    """
    labeled_mask = label(pred_mask, connectivity=2)
    if labeled_mask.max() == 0:
        return pred_mask
    largest_label = labeled_mask == (np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1)
    return largest_label.astype(np.uint8)