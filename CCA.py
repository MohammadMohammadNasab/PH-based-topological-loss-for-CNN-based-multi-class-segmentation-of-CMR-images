import numpy as np
from scipy.ndimage import label

def connected_component_analysis(mask, connectivity=4):
    """
    Retains only the largest connected component in a 2D binary mask.

    :param mask: 2D numpy array (binary mask)
    :param connectivity: 4 for 4-connected, 8 for 8-connected
    :return: 2D numpy array with only the largest component
    """
    # Define connectivity structure
    if connectivity == 4:
        struct = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]])  # 4-connected
    else:
        struct = np.ones((3, 3), dtype=int)  # 8-connected

    # Label connected components
    labeled_mask, num_components = label(mask, structure=struct)

    if num_components == 0:
        return mask  # No components found, return the original mask

    # Find the largest component
    component_sizes = np.bincount(labeled_mask.ravel())[1:]  # Ignore background count (index 0)
    largest_component = np.argmax(component_sizes) + 1  # Component indices start from 1

    # Create a mask with only the largest component
    largest_mask = (labeled_mask == largest_component).astype(np.uint8)

    return largest_mask
