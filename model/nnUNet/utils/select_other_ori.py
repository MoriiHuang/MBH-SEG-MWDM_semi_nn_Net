import os
import shutil

# Paths to your original datasets and re-annotated data
original_datasets = {
    "train": "/home/hcy/FGR_JMRI/Data_FGR_2023/MRI_images/imagesTr",
    "val": "/home/hcy/FGR_JMRI/Data_FGR_2023/MRI_images/imagesTs",
    "test1": "/home/hcy/FGR_JMRI/Data_FGR_2023/MRI_images/imagesTs2",
    "test2": "/home/hcy/FGR_JMRI/Data_FGR_2023/MRI_images/imagesTs3"
}

cur_training_set_path = "/home/hcy/nnUNet/DATASET/nnUNet_raw/Dataset015_FGRhys/imagesTr" 
other_test_set_path = "/home/hcy/nnUNet/DATASET/nnUNet_raw/Dataset015_FGRhys/imagesTs"

# Get the list of filenames in the current training set
training_filenames = {f for f in os.listdir(cur_training_set_path) if f.endswith('.nii.gz')}

# Function to copy non-existing files to the test set path
def copy_missing_files(original_dataset_path, training_filenames, other_test_set_path):
    for file_name in os.listdir(original_dataset_path):
        if file_name.endswith('.nii.gz') and file_name not in training_filenames:
            src_file_path = os.path.join(original_dataset_path, file_name)
            dst_file_path = os.path.join(other_test_set_path, file_name)
            shutil.copy(src_file_path, dst_file_path)
            print(f"Copied {file_name} to {other_test_set_path}")

# Create the other test set path directory if it doesn't exist
os.makedirs(other_test_set_path, exist_ok=True)

# Iterate through all original datasets and copy missing files
for dataset_name, dataset_path in original_datasets.items():
    print(f"Checking {dataset_name} dataset for missing files...")
    copy_missing_files(dataset_path, training_filenames, other_test_set_path)

print("All missing files have been copied.")
