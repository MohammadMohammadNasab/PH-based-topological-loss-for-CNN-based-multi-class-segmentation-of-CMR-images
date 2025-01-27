import numpy as np
import matplotlib.pyplot as plt

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

def plot_nii_slices(image_path, num_slices=5):
    """
    Plot 2D slices from a 3D .nii or .nii.gz file.
    """
    # Load the image
    img = sitk.ReadImage(image_path)
    img_np = sitk.GetArrayFromImage(img)  # Convert to numpy array [Depth, Height, Width]

    # Normalize the image for better visualization
    img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np))
    print(f"Max value in img_np: {np.max(img_np)}")
    print(f"Min value in img_np: {np.min(img_np)}")
    print(img_np.shape)
    # Plot mid-ventricular slices
    mid_slice = img_np.shape[0] // 2
    slices = img_np[mid_slice - num_slices//2 : mid_slice + num_slices//2 + 1]

    # Create a figure to display the slices
    fig, axes = plt.subplots(1, num_slices, figsize=(15, 5))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice, cmap="gray")
        axes[i].set_title(f"Slice {mid_slice - num_slices//2 + i}")

    plt.tight_layout()
    plt.show()

plot_nii_slices(r'C:\Users\Mohammad\Desktop\ML_Project\data\patient101_frame01.nii.gz')
