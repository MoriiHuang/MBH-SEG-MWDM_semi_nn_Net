import os

### Given path of original image (nii.gz) and path of mask (nii.gz), this function will return the difference of files between the two folders(image files xx_0000.nii.gz,mask flies xx.nii.gz), i.e., the files present in the original image folder but not in the mask folder. 
def diff_between_img_mask(original_img_folder, mask_folder):
    # Initialize the list to store the difference
    diff_list = []
    
    # Get the list of files in the original image folder
    img_files = os.listdir(original_img_folder)
    
    # Get the list of files in the mask folder
    mask_files = os.listdir(mask_folder)
    
    # Extract the base names (without extension) from the files
    img_basenames = [os.path.basename(file).replace("_0000.nii.gz", "") for file in img_files]
    mask_basenames = [os.path.basename(file).replace(".nii.gz", "") for file in mask_files]
    # Find the difference between the two lists
    diff_list = list(set(img_basenames) - set(mask_basenames))
    if len(diff_list)==0:
       diff_list = list(set(mask_basenames) - set(img_basenames))
    
    return diff_list


# Example usage
if __name__ == "__main__":
    original_img_folder = "/home/hcy/nnUNet/DATASET/nnUNet_raw/Dataset015_FGRhys/imagesTr"
    mask_folder = "/home/hcy/nnUNet/DATASET/nnUNet_raw/Dataset015_FGRhys/labelsTr"
    
    diff_list = diff_between_img_mask(original_img_folder, mask_folder)
    print(f"Difference between image and mask files: {diff_list}")