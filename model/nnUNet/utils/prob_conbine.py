import os

import SimpleITK
import nibabel as nib
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import torch
import numpy as np
from scipy.ndimage import label
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--overall_mask_folder', type=str, help='overall mask file')
parser.add_argument('--subclass_mask_folder', type=str, help='subclass mask file')
parser.add_argument('--overall_prob_folder', type=str, help='overall prob file')
parser.add_argument('--subclass_prob_folder', type=str, help='subclass prob file')
parser.add_argument('--output_folder', type=str, help='output folder')

def combine_masks(overall_mask_data, subclass_mask_data,overall_prob,sub_prob):
    """
    将整体掩膜和子类掩膜结合在一起，并保存为新的掩膜文件。

    参数:
    overall_mask_file (str): 输入的整体掩膜文件路径。
    subclass_mask_file (str): 输入的子类掩膜文件路径。
    output_file (str): 输出的新掩膜文件路径。
    """
    # 读取整体mask和子类mask文件
    overall_mask_data = np.transpose(overall_mask_data.astype(int),(2,1,0))
    
    subclass_mask_data = np.transpose(subclass_mask_data.astype(int),(2,1,0))

    # 检查加载的数据
    print("Unique values in overall mask:", np.unique(overall_mask_data))
    print("Unique values in subclass mask:", np.unique(subclass_mask_data))

    # 创建新的掩膜数据
    new_mask_data = np.copy(overall_mask_data)
    new_prob_data = np.copy(sub_prob)

    # 重合区域处理：整体Mask重合的部分修改为子类Mask
    new_mask_data[subclass_mask_data > 0] = subclass_mask_data[subclass_mask_data > 0]
    new_prob_data[:,subclass_mask_data > 0] = sub_prob[:,subclass_mask_data >0]

    # 处理只有整体Mask有值而子类Mask没有值的区域
    overall_only_mask = (overall_mask_data > 0) & (subclass_mask_data == 0)
    labeled_array, num_features = label(overall_only_mask)
    
    for i in range(1, num_features + 1):
        region = labeled_array == i
        region_indices = np.argwhere(region)
        
        for index in region_indices:
            x, y, z = index
            window = subclass_mask_data[max(0, x-25):min(subclass_mask_data.shape[0], x+25),
                                        max(0, y-25):min(subclass_mask_data.shape[1], y+25),
                                        max(0, z-25):min(subclass_mask_data.shape[2], z+25)]
            unique, counts = np.unique(window[window > 0], return_counts=True)
            
            if len(counts) > 0:
                majority_label = unique[np.argmax(counts)]
                if majority_label !=1:
                    new_mask_data[x, y, z] = majority_label
                    new_prob_data[majority_label,x, y, z] = overall_prob[1,x, y, z]
                    new_prob_data[:majority_label,x, y, z] = (1-overall_prob[1,x, y, z])/(new_prob_data.shape[0]-1)
                    new_prob_data[majority_label+1:,x, y, z] = (1-overall_prob[1,x, y, z])/(new_prob_data.shape[0]-1)
                if majority_label == 1:
                    if len(counts) > 1:
                        new_mask_data[x, y, z] = unique[np.argsort(counts)[-2]]
                        new_prob_data[unique[np.argsort(counts)[-2]],x, y, z] = overall_prob[1,x, y,z]
                        new_prob_data[:unique[np.argsort(counts)[-2]],x, y, z] = (1-overall_prob[1,x, y, z])/(new_prob_data.shape[0]-1)
                        new_prob_data[unique[np.argsort(counts)[-2]]+1:,x, y, z] = (1-overall_prob[1,x, y, z])/(new_prob_data.shape[0]-1)
                    else:
                        new_mask_data[x, y, z] = 0
                        new_prob_data[1:,x, y, z] = 0
                        new_prob_data[0,x, y, z] = 1
            elif len(counts) == 0:
                new_mask_data[x, y, z] = 0
                new_prob_data[1:,x, y, z] = 0
                new_prob_data[0,x, y, z] = 1

    # 处理子类Mask有值而整体Mask没有值的区域
    subclass_only_mask = (subclass_mask_data > 0) & (overall_mask_data == 0)
    new_mask_data[subclass_only_mask] = subclass_mask_data[subclass_only_mask]
    new_prob_data[:,subclass_only_mask] = sub_prob[:,subclass_only_mask]

    return new_mask_data, new_prob_data


def batch_process(overall_mask_folder, subclass_mask_folder, overall_prob_folder, subclass_prob_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    overall_mask_files = os.listdir(overall_mask_folder)
    for overall_mask_file in overall_mask_files:
        if overall_mask_file.endswith('.nii.gz'):
            prob_name = overall_mask_file.replace('.nii.gz','_0000.npz')
            print(f"Processing {prob_name}")
            overall_mask_data = nib.load(os.path.join(overall_mask_folder, overall_mask_file)).get_fdata()
            subclass_mask_data = nib.load(os.path.join(subclass_mask_folder, overall_mask_file)).get_fdata()
            overall_prob = np.load(os.path.join(overall_prob_folder, prob_name),allow_pickle=True)['probabilities']
            spacing, direction, origin, props = np.load(os.path.join(overall_prob_folder, prob_name),allow_pickle=True)['spacing'], np.load(os.path.join(overall_prob_folder, prob_name),allow_pickle=True)['direction'], np.load(os.path.join(overall_prob_folder, prob_name),allow_pickle=True)['origin'], np.load(os.path.join(overall_prob_folder, prob_name),allow_pickle=True)['props']
            sub_prob = np.load(os.path.join(subclass_prob_folder, prob_name))['probabilities']
            _, new_prob_data = combine_masks(overall_mask_data, subclass_mask_data,overall_prob,sub_prob)
            save_file = os.path.join(output_folder, prob_name)
            np.savez_compressed(save_file, probabilities=new_prob_data,spacing=spacing,direction=direction,origin=origin,props=props)
            print(f"Saved probabilities to {save_file}")

if __name__ == '__main__':
    args = parser.parse_args()
    batch_process(args.overall_mask_folder, args.subclass_mask_folder, args.overall_prob_folder, args.subclass_prob_folder, args.output_folder)

