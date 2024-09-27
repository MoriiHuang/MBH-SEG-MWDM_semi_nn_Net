import os

# count the number of nii files in a folder
def count_nii_files(folder_path):
    # Initialize the count
    count = 0
    
    # Loop through all files in the folder
    for file in os.listdir(folder_path):
        # Check if the file is a nii file
        if file.endswith(".nii") or file.endswith(".nii.gz"):
            # Increment the count
            count += 1
    
    return count

# Example usage
if __name__ == "__main__":
    folder_path = "/home/hcy/nnUNet/DATASET/nnUNet_raw/Dataset015_FGRhys/imagesTr"
    nii_count = count_nii_files(folder_path)
    print(f"Number of nii files in {folder_path}: {nii_count}")