import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import cv2


def dura_window(img):
    return sigmoid_window(img, 300, 500)  # 窗位=300, 窗宽=500

def map_to_gradient_sig(grey_img):
    rainbow_img = np.zeros((grey_img.shape[0], grey_img.shape[1], 3))
    rainbow_img[:, :, 0] = np.clip(4*grey_img - 2, 0, 1.0) * (grey_img > 0.01) * (grey_img <= 1.0)
    rainbow_img[:, :, 1] =  np.clip(4*grey_img * (grey_img <=0.75), 0,1) + np.clip((-4*grey_img + 4) * (grey_img > 0.75), 0, 1)
    rainbow_img[:, :, 2] = np.clip(-4*grey_img + 2, 0, 1.0) * (grey_img > 0.01) * (grey_img <= 1.0)
    return rainbow_img

def sigmoid_window(img, window_center, window_width, U=1.0, eps=(1.0 / 255.0)):

    ue = np.log((U / eps) - 1.0)
    W = (2 / window_width) * ue
    b = ((-2 * window_center) / window_width) * ue
    z = W * img + b
    img = U / (1 + np.power(np.e, -1.0 * z))
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img

def sigmoid_bsb_window(img):
    brain_img = sigmoid_window(img, 40, 80)
    subdural_img = sigmoid_window(img, 80, 200)
    bone_img = sigmoid_window(img, 600, 2000)
    combo = (brain_img*0.35+subdural_img*0.5+bone_img*0.15)
    combo_norm = (combo - np.min(combo)) / (np.max(combo) - np.min(combo))
    return map_to_gradient_sig(combo_norm)

def sigmoid_bsb_window_without_gradient(img):
    brain_img = sigmoid_window(img, 40, 80)
    subdural_img = sigmoid_window(img, 80, 200)
    bone_img = sigmoid_window(img, 600, 2000)
    combo = np.zeros((img.shape[0], img.shape[1], 3))
    combo[:, :, 0] = brain_img
    combo[:, :, 1] = subdural_img
    combo[:, :, 2] = bone_img
    combo_norm = (combo - np.min(combo)) / (np.max(combo) - np.min(combo))
    return combo_norm


def process_nii_ct(input_nii_path, output_nii_path):
    # 加载NIfTI文件
    print(f"Processing {input_nii_path}")
    nii_img = nib.load(input_nii_path)
    nii_data = nii_img.get_fdata()
    qform = nii_img.get_qform()
    sform = nii_img.get_sform()

    # 获取原始数据的形状
    original_shape = nii_data.shape

    # 初始化一个空数组，用于存储处理后的数据
    processed_data = np.zeros_like(nii_data)

    # 遍历每个切片，应用 sigmoid_advanced_window 处理
    for i in tqdm(range(original_shape[2]), desc="Processing slices"):
        slice_data = nii_data[:, :, i]
        processed_slice = sigmoid_bsb_window(slice_data)
        # save as png
        processed_data[:, :, i] = processed_slice.mean(axis=2)

    # 重新将处理后的数据保存为新的NIfTI文件
    processed_img = nib.Nifti1Image(processed_data, affine=nii_img.affine)
    processed_img.set_qform(qform)
    processed_img.set_sform(sform)
    nib.save(processed_img, output_nii_path)
    print(f"Processed NIfTI file saved as {output_nii_path}")

def process_nii_ct_mul(input_nii_path, output_nii_path):
    # 加载NIfTI文件
    print(f"Processing {input_nii_path}")
    nii_img = nib.load(input_nii_path)
    nii_data = nii_img.get_fdata()
    qform = nii_img.get_qform()
    sform = nii_img.get_sform()

    # 获取原始数据的形状
    original_shape = nii_data.shape

    # 初始化一个空数组，用于存储处理后的数据
    processed_data_r = np.zeros_like(nii_data)
    processed_data_g = np.zeros_like(nii_data)
    processed_data_b = np.zeros_like(nii_data)
    # if not os.path.exists(f"/home/hcy/nnUNet/DATASET/nnUNet_raw/FirstStage_test_Mul/imagespng/{output_nii_path.split('/')[-1].replace('.nii.gz','')}"):
    #     os.makedirs(f"/home/hcy/nnUNet/DATASET/nnUNet_raw/FirstStage_test_Mul/imagespng/{output_nii_path.split('/')[-1].replace('.nii.gz','')}")

    # 遍历每个切片，应用 sigmoid_advanced_window 处理
    for i in tqdm(range(original_shape[2]), desc="Processing slices"):
        slice_data = nii_data[:, :, i]
        processed_slice = sigmoid_bsb_window_without_gradient(slice_data)
        processed_data_r[:, :, i] = processed_slice[:, :, 0]
        processed_data_g[:, :, i] = processed_slice[:, :, 1]
        processed_data_b[:, :, i] = processed_slice[:, :, 2]
        # # ratota 270 degree clockwise and save as png
        # processed_slice = cv2.rotate(processed_slice, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # cv2.imwrite(f"/home/hcy/nnUNet/DATASET/nnUNet_raw/FirstStage_test_Mul/imagespng/{output_nii_path.split('/')[-1].replace('.nii.gz','')}/{i}.png", (processed_slice*255).astype(np.uint8))


    # 重新将处理后的数据保存为新的NIfTI文件
    processed_img_r = nib.Nifti1Image(processed_data_r, affine=nii_img.affine)
    processed_img_r.set_qform(qform)
    processed_img_r.set_sform(sform)
    nib.save(processed_img_r, output_nii_path)
    processed_img_g = nib.Nifti1Image(processed_data_g, affine=nii_img.affine)
    processed_img_g.set_qform(qform)
    processed_img_g.set_sform(sform)
    nib.save(processed_img_g, output_nii_path.replace("0000", "0001"))
    processed_img_b = nib.Nifti1Image(processed_data_b, affine=nii_img.affine)
    processed_img_b.set_qform(qform)
    processed_img_b.set_sform(sform)
    nib.save(processed_img_b, output_nii_path.replace("0000", "0002"))
    print(f"Processed NIfTI file saved as {output_nii_path}")


def batch_process_nii_ct(input_dir, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录下的所有NIfTI文件
    for nii_file in os.listdir(input_dir):
        if nii_file.endswith(".nii.gz"):
            input_nii_path = os.path.join(input_dir, nii_file)
            output_nii_path = os.path.join(output_dir, nii_file.replace(".nii.gz", "_0000.nii.gz"))
            try:
                # process_nii_ct(input_nii_path, output_nii_path)
                process_nii_ct_mul(input_nii_path, output_nii_path)
                # # process_nii_ct_s(input_nii_path, output_nii_path)
                # process_nii_ct_b(input_nii_path, output_nii_path)
            except Exception as e:
                print(f"Error processing {input_nii_path}: {e}")
                continue

if __name__ == "__main__":
    input_folder = "/home/hcy/label_192/images"
    output_nii_path = "/home/hcy/nnUNet/DATASET/nnUNet_raw/Dataset011_MBHMul/imagesTr"
    batch_process_nii_ct(input_folder, output_nii_path)