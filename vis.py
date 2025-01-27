import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def rotate_pair(image, label, angle):
    """
    Rotate both image and label mask by specified angle.
    image: input image
    label: label mask
    angle: rotation angle in degrees
    """
    # Rotate image with interpolation
    rotated_image = ndimage.rotate(image, angle, reshape=False)
    # Rotate label with nearest neighbor interpolation to preserve label values
    rotated_label = ndimage.rotate(label, angle, reshape=False, order=0)
    
    return rotated_image, rotated_label

def random_crop_pair(image, label, crop_size):
    """
    Randomly crop image and label with the same coordinates.
    Handles cases where input is smaller than crop_size using padding.
    
    image: input image
    label: label mask
    crop_size: tuple of (height, width) for desired crop size
    """
    if not isinstance(crop_size, tuple):
        crop_size = (crop_size, crop_size)
        
    # Get current sizes
    img_h, img_w = image.shape
    crop_h, crop_w = crop_size
    
    # If image is smaller than crop size, pad it
    if img_h < crop_h or img_w < crop_w:
        pad_h = max(crop_h - img_h, 0)
        pad_w = max(crop_w - img_w, 0)
        
        # Pad image with zeros
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant')
        # Pad label with zeros (background)
        label = np.pad(label, ((0, pad_h), (0, pad_w)), mode='constant')
        
        img_h, img_w = image.shape
    
    # Generate random valid crop coordinates
    h_start = np.random.randint(0, img_h - crop_h + 1)
    w_start = np.random.randint(0, img_w - crop_w + 1)
    
    # Crop both image and label
    cropped_image = image[h_start:h_start+crop_h, w_start:w_start+crop_w]
    cropped_label = label[h_start:h_start+crop_h, w_start:w_start+crop_w]
    
    return cropped_image, cropped_label

# Define the file paths
image_file = r"C:\Users\Mohammad\Desktop\ML_Project\data\preprocessed\train\patient001\frame01_slice039_image.npy"
label_file = r"C:\Users\Mohammad\Desktop\ML_Project\data\preprocessed\train\patient001\frame01_slice039_label.npy"

# Load the files
image = np.load(image_file)
label = np.load(label_file)

# Rotate image and label by 45 degrees
rotated_image, rotated_label = rotate_pair(image, label, 45)

# Apply random crop (e.g., 128x128)
cropped_image, cropped_label = random_crop_pair(rotated_image, rotated_label, (128, 128))

# Normalize image after rotation and cropping
cropped_image = (cropped_image - np.min(cropped_image)) / (np.max(cropped_image) - np.min(cropped_image))

# Print the max and min values in the image
print("Max value in image:", np.max(cropped_image))
print("Min value in image:", np.min(cropped_image))

# Plot the image and label
plt.figure(figsize=(12, 6))

# Plot the image
plt.subplot(1, 2, 1)
plt.title("Cropped & Rotated Image")
plt.imshow(cropped_image, cmap='gray')
plt.colorbar()
plt.axis('off')

# Plot the label
plt.subplot(1, 2, 2)
plt.title("Cropped & Rotated Label")
plt.imshow(cropped_label, cmap='viridis')
plt.colorbar()
plt.axis('off')

# Show the plots
plt.tight_layout()
plt.show()
