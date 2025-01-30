import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

def resample_image(image, target_spacing, is_label=False):
    """
    Resample an image to target spacing.
    Handles metadata and invalid values.
    """
    # Get original spacing, size, and direction
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    original_direction = image.GetDirection()
    original_origin = image.GetOrigin()

    # Calculate new size
    new_size = [
        int(np.round(original_size[0] * (original_spacing[0] / target_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / target_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / target_spacing[2])))
    ]

    # Create resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(original_direction)
    resampler.SetOutputOrigin(original_origin)
    resampler.SetTransform(sitk.Transform())

    # Set interpolation and background value
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # NN for labels
        resampler.SetDefaultPixelValue(0)  # Background value for labels
    else:
        resampler.SetInterpolator(sitk.sitkLinear)  # Linear for images
        resampler.SetDefaultPixelValue(-1024)  # Common background value for CT/MR

    # Execute resampling
    resampled_image = resampler.Execute(image)

    return resampled_image

def preprocess_case(image_path, label_path, output_dir, target_spacing=(1.25, 1.25, 1.25)):
    """
    Preprocess a single CMR case and save slices with encoded filenames.
    """
    # Load image and label
    img = sitk.ReadImage(image_path)
    lbl = sitk.ReadImage(label_path)

    # Resample image and label
    img_resampled = resample_image(img, target_spacing, is_label=False)
    lbl_resampled = resample_image(lbl, target_spacing, is_label=True)

    # Convert to numpy arrays
    img_np = sitk.GetArrayFromImage(img_resampled)  # [Depth, H, W]
    lbl_np = sitk.GetArrayFromImage(lbl_resampled)

    # Handle invalid values (NaN or inf)
    img_np = np.nan_to_num(img_np, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN/inf with 0

    # Extract mid-ventricular slices
    mid_slice = img_np.shape[0] // 2
    img_slices = img_np[mid_slice-1:mid_slice+2]  # 3 slices
    lbl_slices = lbl_np[mid_slice-1:mid_slice+2]

    # Normalize image slices (zero mean, unit variance)
    mask = img_slices > 0  # Mask for valid regions
    if np.any(mask):
        img_slices[mask] = (img_slices[mask] - np.mean(img_slices[mask])) / np.std(img_slices[mask])

    # Save slices with encoded filenames
    patient_id = os.path.basename(image_path).split("_frame")[0]
    frame = os.path.basename(image_path).split("_frame")[1].split(".nii.gz")[0]

    # Create patient-specific output directory
    patient_dir = os.path.join(output_dir, patient_id)
    os.makedirs(patient_dir, exist_ok=True)

    for i in range(img_slices.shape[0]):
        slice_idx = mid_slice - 1 + i  # Original slice index
        slice_str = f"slice{slice_idx:03d}"
        
        # Create filenames
        img_filename = f"frame{frame}_{slice_str}_image.npy"
        lbl_filename = f"frame{frame}_{slice_str}_label.npy"
        
        # Save to patient directory
        np.save(os.path.join(patient_dir, img_filename), img_slices[i])
        np.save(os.path.join(patient_dir, lbl_filename), lbl_slices[i])
# Example usage
import glob
data_dir = 'data/database/train'

# Create output directory change as you want
output_dir = "data/preprocessed/train"

os.makedirs(output_dir, exist_ok=True)

# Get all image files (adjust path as needed)
label_files = sorted(glob.glob(f"{data_dir}/patient*/patient*_frame*_gt.nii.gz"))
image_files = [f.replace("_gt", "") for f in label_files]
print(f'Length of label files: {len(label_files)}')
assert len(label_files) == len(image_files)

print(f'sample image files: {image_files[:2]}')
print(f"sample label files: {label_files[:2]}")
# Process all cases
for img_path, lbl_path in tqdm(zip(image_files, label_files), total= len(image_files)):
    preprocess_case(
        image_path=img_path,
        label_path=lbl_path,
        output_dir=output_dir
    )