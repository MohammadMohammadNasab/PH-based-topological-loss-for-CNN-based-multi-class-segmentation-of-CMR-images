from scipy import ndimage
import torch
from torch.utils.data import Sampler, Dataset
import os
import random
import numpy as np
from torchvision import transforms
def get_patient_data(data_dir):
    """
    Returns a dictionary where keys are patient IDs and values are lists of image/label paths
    for each patient.
    """
    patient_data = {}
    all_image_paths = []
    all_lbl_paths = []
    # Iterate through patient directories
    for patient_dir in os.listdir(data_dir):
        patient_folder = os.path.join(data_dir, patient_dir)
        
        if os.path.isdir(patient_folder):
            image_paths = sorted([os.path.join(patient_folder, f) for f in os.listdir(patient_folder) if f.endswith('_image.npy')])
            label_paths = sorted([os.path.join(patient_folder, f) for f in os.listdir(patient_folder) if f.endswith('_label.npy')])
            all_image_paths.extend(image_paths)
            all_lbl_paths.extend(label_paths)
            if image_paths and label_paths:
                patient_data[patient_dir] = list(zip(image_paths, label_paths))
                
    return patient_data, all_image_paths, all_lbl_paths



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


class TrainACDCDataset(Dataset):
    def __init__(self, data_dict):
        # data_dict is the dictionary returned by get_patient_data
        self.data_dict = data_dict
        self.patient_ids = list(self.data_dict.keys())

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        # Get all (image, label) pairs for this patient, then pick one slice
        slices = self.data_dict[patient_id]
        img_path, lbl_path = random.choice(slices)

        image = np.load(img_path)
        label = np.load(lbl_path)
        
        # Rotate image and label by a random angle between -30 and 30 degrees
        angle = random.uniform(-30, 30)
        rotated_image, rotated_label = rotate_pair(image, label, angle)

        # Apply random crop (e.g., 128x128)
        cropped_image, cropped_label = random_crop_pair(rotated_image, rotated_label, (224, 224))
        # Normalize image after rotation and cropping
        cropped_image = (cropped_image - np.min(cropped_image)) / (np.max(cropped_image) - np.min(cropped_image))


        # Convert the image to a PyTorch tensor
        cropped_image_tensor = torch.tensor(cropped_image, dtype=torch.float32)

       
        cropped_image_tensor = cropped_image_tensor.unsqueeze(0)

        # Convert the label to a PyTorch tensor
        cropped_label_tensor = torch.tensor(cropped_label, dtype=torch.long)


        return cropped_image_tensor, cropped_label_tensor



class ValACDCDataset(Dataset):
    def __init__(self, images, labels):
        # data_dict is the dictionary returned by get_patient_data
        self.images_paths = images
        self.label_paths = labels

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        img_path = self.images_paths[idx]
        lbl_path = self.label_paths[idx]

        image = np.load(img_path)
        label = np.load(lbl_path)
        
        # Rotate image and label by a random angle between -30 and 30 degrees
        angle = random.uniform(-30, 30)
        rotated_image, rotated_label = rotate_pair(image, label, angle)

        # Apply random crop (e.g., 128x128)
        cropped_image, cropped_label = random_crop_pair(rotated_image, rotated_label, (224, 224))
        # Normalize image after rotation and cropping
        cropped_image = (cropped_image - np.min(cropped_image)) / (np.max(cropped_image) - np.min(cropped_image))


        # Convert the image to a PyTorch tensor
        cropped_image_tensor = torch.tensor(cropped_image, dtype=torch.float32)

        
        cropped_image_tensor = cropped_image_tensor.unsqueeze(0)

        # Convert the label to a PyTorch tensor
        cropped_label_tensor = torch.tensor(cropped_label, dtype=torch.long)


        return cropped_image_tensor, cropped_label_tensor
