### 3 model prediction voting

import os
import numpy as np
from scipy.ndimage import label
import nibabel as nib

import argparse

parser = argparse.ArgumentParser(description=' models prediction voting')

parser.add_argument('--mask_folders', type=str, help='mask folders')
parser.add_argument('--output_folder', type=str, help='output folder')
parser.add_argument('--mode', type=str, default='vote_major', help='vote or vote_major')

### give example in shell command


### 3 model if any of the model predict the label, then the final label is the label with the highest vote
def vote(mask_lists):
    mask_data = []
    for mask in mask_lists:
        mask_data.append(nib.load(mask).get_fdata().astype(int))
        # print(nib.load(mask).get_fdata().astype(int).shape)
    new_mask_data = np.zeros_like(mask_data[0])
    mask_data = np.array(mask_data)
    # Reshape the mask data to a 2D array for easier processing
    reshaped_mask_data = mask_data.reshape(len(mask_data), -1)

    # Count the occurrences along the 0th axis, ignoring zeros
    mode_result = np.array([np.bincount(reshaped_mask_data[:, i][reshaped_mask_data[:, i] != 0], minlength=len(mask_data)).argmax() 
                            for i in range(reshaped_mask_data.shape[1])])

    # Reshape the result back to the original 3D shape
    new_mask_data = mode_result.reshape(mask_data[0].shape)

    del mask_data

    return new_mask_data

def vote_majority(mask_lists):
    mask_data = []
    for mask in mask_lists:
        mask_data.append(nib.load(mask).get_fdata().astype(int))
    
    mask_data = np.array(mask_data)
    new_mask_data = np.zeros_like(mask_data[0])

    # Reshape the mask data to a 2D array for easier processing
    reshaped_mask_data = mask_data.reshape(len(mask_data), -1)

    # Calculate the majority threshold
    majority_threshold = len(mask_data) // 2 + 1

    # Apply majority voting: only assign a label if it has majority support
    def majority_vote(values):
        # Count occurrences ignoring zeros
        counts = np.bincount(values[values != 0], minlength=len(mask_data))
        # Find the label that meets the majority threshold
        majority_label = np.where(counts >= majority_threshold)[0]
        if len(majority_label) > 0:
            return majority_label[0]
        else:
            return 0

    mode_result = np.array([majority_vote(reshaped_mask_data[:, i]) for i in range(reshaped_mask_data.shape[1])])

    # Reshape the result back to the original 3D shape
    new_mask_data = mode_result.reshape(mask_data[0].shape)

    del mask_data

    return new_mask_data
def batch_process(input_folders,output_folder,mode='vote'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    mask_list = os.listdir(input_folders[0])

    for mask in mask_list:
        mask_paths = []
        for folder in input_folders:
            mask_path = os.path.join(folder,mask)
            print(f'加载掩膜 {mask_path}','非零体素值:',np.count_nonzero(nib.load(mask_path).get_fdata()))
            mask_paths.append(mask_path)
        if mode == 'vote':
            new_mask_data = vote(mask_paths)
        elif mode == 'vote_major':
            new_mask_data = vote_majority(mask_paths)
        new_mask_img = nib.Nifti1Image(new_mask_data, nib.load(mask_paths[0]).affine, nib.load(mask_paths[0]).header)
        nib.save(new_mask_img, os.path.join(output_folder,mask))
        print(f'结合后的掩膜已保存到 {os.path.join(output_folder,mask)}','非零体素值:',np.count_nonzero(new_mask_data))

if __name__ == "__main__":
    args = parser.parse_args()
    batch_process(args.mask_folders.split(),args.output_folder,args.mode)