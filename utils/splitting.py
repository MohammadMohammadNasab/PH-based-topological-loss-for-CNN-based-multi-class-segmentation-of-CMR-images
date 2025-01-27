import random
import os

source_dir = "data/preprocessed/test_and_val"
test_dir = "data/preprocessed/test"
val_dir = "data/preprocessed/val"

# Create destination directories if they don't exist
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get a list of patient subfolders
patient_folders = [f for f in os.listdir(source_dir) if f.startswith("patient")]

# Shuffle the patient folders randomly
random.shuffle(patient_folders)

# Calculate the split index
split_index = len(patient_folders) // 2

# Split the patient folders into test and validation sets
test_patients = patient_folders[:split_index]
val_patients = patient_folders[split_index:]

# Move the patient folders to the respective destination directories
for patient in test_patients:
    source_path = os.path.join(source_dir, patient)
    destination_path = os.path.join(test_dir, patient)
    shutil.move(source_path, destination_path)

for patient in val_patients:
    source_path = os.path.join(source_dir, patient)
    destination_path = os.path.join(val_dir, patient)
    shutil.move(source_path, destination_path)