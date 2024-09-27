import os
import nibabel as nib
import numpy as np
from scipy.ndimage import label

### make input by argparse
import argparse
parser = argparse.ArgumentParser(description='Combine overall mask and subclass mask')
parser.add_argument('overall_mask_folder', type=str, help='Path to the overall mask file')
parser.add_argument('subclass_mask_folder', type=str, help='Path to the subclass mask file')
parser.add_argument('output_folder', type=str, help='Path to the output file')
args = parser.parse_args()


# example
# python combine.py /home/hcy/nnUNet/DATASET/nnUNet_inference/Dataset008_MBHMerge/1.nii.gz /home/hcy/nnUNet/DATASET/nnUNet_inference/Dataset003_MBH/1.nii.gz /home/hcy/nnUNet/DATASET/nnUNet_inference/combine/1.nii.gz 

def combine_masks(overall_mask_file, subclass_mask_file, output_file):
    """
    将整体掩膜和子类掩膜结合在一起，并保存为新的掩膜文件。

    参数:
    overall_mask_file (str): 输入的整体掩膜文件路径。
    subclass_mask_file (str): 输入的子类掩膜文件路径。
    output_file (str): 输出的新掩膜文件路径。
    """
    # 读取整体mask和子类mask文件
    overall_mask_img = nib.load(overall_mask_file)
    overall_mask_data = overall_mask_img.get_fdata().astype(int)
    
    subclass_mask_img = nib.load(subclass_mask_file)
    subclass_mask_data = subclass_mask_img.get_fdata().astype(int)

    # 检查加载的数据
    print("Unique values in overall mask:", np.unique(overall_mask_data))
    print("Unique values in subclass mask:", np.unique(subclass_mask_data))

    # 创建新的掩膜数据
    new_mask_data = np.copy(overall_mask_data)

    # 重合区域处理：整体Mask重合的部分修改为子类Mask
    new_mask_data[subclass_mask_data > 0] = subclass_mask_data[subclass_mask_data > 0]

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
                if majority_label == 1:
                    if len(counts) > 1:
                        new_mask_data[x, y, z] = unique[np.argsort(counts)[-2]]
                    else:
                        new_mask_data[x, y, z] = 0
            elif len(counts) == 0:
                new_mask_data[x, y, z] = 0

    # 处理子类Mask有值而整体Mask没有值的区域
    subclass_only_mask = (subclass_mask_data > 0) & (overall_mask_data == 0)
    new_mask_data[subclass_only_mask] = subclass_mask_data[subclass_only_mask]

    # 创建新的Nifti1Image对象
    new_mask_img = nib.Nifti1Image(new_mask_data, overall_mask_img.affine, overall_mask_img.header)

    # 保存新的掩膜文件
    nib.save(new_mask_img, output_file)
    print(f'结合后的掩膜已保存到 {output_file}')

def batch_process(overall_mask_folder, subclass_mask_folder, output_folder):
    """
    批量处理整体掩膜和子类掩膜，并保存结果到指定文件夹。

    参数:
    overall_mask_folder (str): 整体掩膜文件夹路径。
    subclass_mask_folder (str): 子类掩膜文件夹路径。
    output_folder (str): 输出文件夹路径。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    overall_files = {f for f in os.listdir(overall_mask_folder) if f.endswith('.nii.gz')}
    subclass_files = {f for f in os.listdir(subclass_mask_folder) if f.endswith('.nii.gz')}
    
    common_files = overall_files & subclass_files
    
    for file_name in common_files:
        overall_mask_file = os.path.join(overall_mask_folder, file_name)
        subclass_mask_file = os.path.join(subclass_mask_folder, file_name)
        print(overall_mask_file, subclass_mask_file)
        output_file = os.path.join(output_folder, file_name)
        
        combine_masks(overall_mask_file, subclass_mask_file, output_file)

# # 示例用法
# overall_mask_folder = '/home/hcy/nnUNet/DATASET/nnUNet_inference/Dataset008_MBHMerge'  # 替换为你的整体掩膜文件夹路径
# subclass_mask_folder = '/home/hcy/nnUNet/DATASET/nnUNet_inference/Dataset003_MBH'  # 替换为你的子类掩膜文件夹路径
# output_folder = '/home/hcy/nnUNet/DATASET/nnUNet_inference/combine'  # 替换为你的输出文件夹路径
if __name__ == "__main__":
    # combine_masks('/home/hcy/nnUNet/DATASET/nnUNet_inference/nnsam/FirstStage_test_3/ID_0bee00a2_ID_0b9e78b135.nii.gz','/home/hcy/nnUNet/DATASET/nnUNet_inference/AugSeg/FirstStage_test_6/ID_0bee00a2_ID_0b9e78b135.nii.gz','test.nii.gz')
    batch_process(args.overall_mask_folder, args.subclass_mask_folder, args.output_folder)
