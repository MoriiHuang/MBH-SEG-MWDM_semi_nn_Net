import SimpleITK as sitk
import numpy as np
import os
import argparse

def process_image(input_image_path, output_image_path):
    # 读取原始图像
    original_image = sitk.ReadImage(input_image_path)
    processed_image = sitk.ReadImage(output_image_path)
    processed_array = sitk.GetArrayFromImage(processed_image)
    # 获取原始图像的元数据信息
    original_spacing = original_image.GetSpacing()
    original_origin = original_image.GetOrigin()
    original_direction = original_image.GetDirection()

    # 将numpy数组转换回图像
    processed_image = sitk.GetImageFromArray(processed_array)

    # 设置原始的元数据信息
    processed_image.SetSpacing(original_spacing)
    processed_image.SetOrigin(original_origin)
    processed_image.SetDirection(original_direction)

    # 保存处理后的图像
    sitk.WriteImage(processed_image, output_image_path)

    ### 将output_image_path 重命名，去掉_nyul
    os.rename(output_image_path, output_image_path.replace('_nyul',''))

def main(input_dir, output_dir):
    # 获取输入目录中的所有文件
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.nii.gz'):  # 根据需要调整文件扩展名
                input_image_path = os.path.join(root, file)
                output_image_path = os.path.join(output_dir, file.replace('.nii.gz', '_nyul.nii.gz'))
                process_image(input_image_path, output_image_path)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch process MRI images.')
    parser.add_argument('--input', required=True, help='Input directory containing original images.')
    parser.add_argument('--output', required=True, help='Output directory for processed images.')
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)

    main(args.input, args.output)
