import os
import shutil

# Paths to your original datasets and re-annotated data
original_datasets = {
    "train": "/home/hcy/FGR_JMRI/Data_FGR_2023/MRI_images/imagesTr",
    "val": "/home/hcy/FGR_JMRI/Data_FGR_2023/MRI_images/imagesTs",
    "test1": "/home/hcy/FGR_JMRI/Data_FGR_2023/MRI_images/imagesTs2",
    "test2": "/home/hcy/FGR_JMRI/Data_FGR_2023/MRI_images/imagesTs3"
}

# Path to the re-annotated data
re_annotated_data_path = "/home/hcy/FGR_JMRI/hys_label"

# Path to save the new training set
new_training_set_path = "/home/hcy/nnUNet/DATASET/nnUNet_raw/Dataset010_FGRhys/imagesTs"
os.makedirs(new_training_set_path, exist_ok=True)

def extract_images_based_on_annotation(re_annotated_file, original_data_folder, save_folder):
    # Extract the base name (without extension) from the re-annotated file
    re_annotated_basename = os.path.basename(re_annotated_file).replace(".nii.gz", "")
    
    # Construct the corresponding image file name
    corresponding_image_name = f"{re_annotated_basename}_0000.nii.gz"
    corresponding_image_path = os.path.join(original_data_folder, corresponding_image_name)

    # Check if the corresponding image exists
    if os.path.exists(corresponding_image_path):
        # Copy the original image to the new training set folder
        print(f"Copied: {corresponding_image_name}")
        shutil.copy(corresponding_image_path, os.path.join(save_folder, corresponding_image_name))
    else:
        print(f"Image not found for annotation: {re_annotated_basename}")
def process_datasets(original_datasets, re_annotated_data_path, new_training_set_path):
    # Process each dataset and extract corresponding original images

    for re_annotated_file in os.listdir(re_annotated_data_path):
        if re_annotated_file.endswith(".nii.gz"):
            re_annotated_file_path = os.path.join(re_annotated_data_path, re_annotated_file)
            for dataset_name, dataset_path in original_datasets.items():
                print(f"Processing dataset: {dataset_name}","dataset_path:",dataset_path)
                extract_images_based_on_annotation(re_annotated_file_path, dataset_path, new_training_set_path)
            

if __name__ == "__main__":
    # Call the function to process datasets
    process_datasets(original_datasets, re_annotated_data_path, new_training_set_path)

    print("Extraction complete. New training set created.")