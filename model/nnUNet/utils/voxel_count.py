import os
import nibabel as nib
import numpy as np
import collections
import argparse

def count_voxels_by_class(mask_data):
    """统计每个类别的体素数量，并忽略类别0."""
    voxel_counts = collections.Counter(mask_data.flatten())
    if 0 in voxel_counts:
        del voxel_counts[0]  # 忽略类别0
    return dict(voxel_counts)

def process_folder(input_folder_path, output_file_path):
    """处理文件夹中的所有nii.gz文件，并将结果汇总到一个txt文件中."""
    class_totals = collections.defaultdict(int)  # 用于存储每个类别的总计
    class_details = collections.defaultdict(dict)  # 用于存储每个类别的详细信息

    # 遍历文件夹中的每个nii.gz文件
    for filename in os.listdir(input_folder_path):
        if filename.endswith('.nii.gz'):
            file_path = os.path.join(input_folder_path, filename)
            # 读取nii.gz文件
            img = nib.load(file_path)
            mask_data = img.get_fdata()

            # 统计体素值
            voxel_counts = count_voxels_by_class(mask_data)

            # 更新总计和详细信息
            for voxel_value, count in voxel_counts.items():
                class_totals[voxel_value] += count
                class_details[voxel_value][filename] = count

    # 写入统计结果到文件
    with open(output_file_path, 'w') as output_file:
        # 写入总的统计结果
        output_file.write("Overall Voxel Counts by Class:\n")
        for voxel_value, total_count in sorted(class_totals.items()):
            output_file.write(f"Class {int(voxel_value)}: {total_count} voxels\n")
            output_file.write(f"Details: {class_details[voxel_value]}\n")
        output_file.write("=" * 40 + "\n")

        # 写入每个文件的详细统计结果
        output_file.write("Voxel Counts per File:\n")
        for filename in os.listdir(input_folder_path):
            if filename.endswith('.nii.gz'):
                output_file.write(f"File: {filename}\n")
                file_voxel_counts = {k: v[filename] for k, v in class_details.items() if filename in v}
                for voxel_value, count in file_voxel_counts.items():
                    output_file.write(f"Class {int(voxel_value)}: {count} voxels\n")
                output_file.write("-" * 40 + "\n")

    print(f"Statistics for all files saved to {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Voxel count for nnUNet')
    parser.add_argument('--input_folder', type=str, default='/home/hcy/nnUNet/DATASET/nnUNet_inference/merge/nntrans_AugSeg_nnSAM_bsb', help='Path to the input folder')
    parser.add_argument('--output_folder', type=str, default='path/to/output/folder', help='Path to the output folder')
    parser.add_argument('--output_file', type=str, required=False, help='Path to the output')
    args = parser.parse_args()
    input_folder_path = args.input_folder
    output_folder_path = args.output_folder
    if args.output_file:
        file_name = args.output_file
    else:
        file_name = input_folder_path.split('/')[-1]
    os.makedirs(output_folder_path, exist_ok=True)
    output_path = os.path.join(output_folder_path, f"{file_name}_voxel_count.txt")
    
    process_folder(input_folder_path, output_path)
