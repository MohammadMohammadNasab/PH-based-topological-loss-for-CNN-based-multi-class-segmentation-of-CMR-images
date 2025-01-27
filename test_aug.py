import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.dataloading import RandomRotation, RandomCrop, MyCompose
import torch
from torchvision import transforms

# Load a single slice for testing
img_path = r"C:\Users\Mohammad\Desktop\ML_Project\data\preprocessed\val\patient106\frame01_slice035_image.npy"
lbl_path = r"C:\Users\Mohammad\Desktop\ML_Project\data\preprocessed\val\patient106\frame01_slice035_label.npy"

# Load and convert to PIL
image = Image.fromarray(np.load(img_path))
label = Image.fromarray(np.load(lbl_path))

# Create a figure to show multiple transform combinations
plt.figure(figsize=(15, 10))

# Original images
plt.subplot(3, 4, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(3, 4, 2)
plt.imshow(label, cmap='tab20')
plt.title('Original Label')
plt.axis('off')

# Test Random Rotation
rotation_transform = MyCompose([RandomRotation(degrees=30)])
rot_img, rot_lbl = rotation_transform(image, label)

plt.subplot(3, 4, 3)
plt.imshow(rot_img, cmap='gray')
plt.title('Rotation 30°')
plt.axis('off')

plt.subplot(3, 4, 4)
plt.imshow(rot_lbl, cmap='tab20')
plt.title('Rotated Label')
plt.axis('off')

# Test Random Crop
crop_transform = MyCompose([RandomCrop(size=(256, 256))])
crop_img, crop_lbl = crop_transform(image, label)

plt.subplot(3, 4, 5)
plt.imshow(crop_img, cmap='gray')
plt.title('Crop 256x256')
plt.axis('off')

plt.subplot(3, 4, 6)
plt.imshow(crop_lbl, cmap='tab20')
plt.title('Cropped Label')
plt.axis('off')

# Test Combined transforms
combined_transform = MyCompose([
    RandomRotation(degrees=30),
    RandomCrop(size=(256, 256))
])

combined_img, combined_lbl = combined_transform(image, label)

# Normalize the image
normalize_transform = transforms.Normalize(mean=[0.5], std=[0.5])
normalized_img = normalize_transform(transforms.ToTensor()(combined_img)).numpy().transpose(1, 2, 0)
normalized_lbl = combined_lbl  # Assuming label normalization is not required

plt.subplot(3, 4, 7)
plt.imshow(normalized_img, cmap='gray')
plt.title('Normalized Image')
plt.axis('off')

plt.subplot(3, 4, 8)
plt.imshow(normalized_lbl, cmap='tab20')
plt.title('Normalized Label')
plt.axis('off')
plt.subplot(3, 4, 7)
plt.imshow(combined_img, cmap='gray')
plt.title('Rotation + Crop')
plt.axis('off')

plt.subplot(3, 4, 8)
plt.imshow(combined_lbl, cmap='tab20')
plt.title('Combined Label')
plt.axis('off')

# Test another combination with different parameters
combined_transform2 = MyCompose([
    RandomRotation(degrees=45),
    RandomCrop(size=(224, 224))
])

combined_img2, combined_lbl2 = combined_transform2(image, label)

plt.subplot(3, 4, 9)
plt.imshow(combined_img2, cmap='gray')
plt.title('Rot45° + Crop224')
plt.axis('off')

plt.subplot(3, 4, 10)
plt.imshow(combined_lbl2, cmap='tab20')
plt.title('Combined Label 2')
plt.axis('off')

plt.tight_layout()
plt.show()