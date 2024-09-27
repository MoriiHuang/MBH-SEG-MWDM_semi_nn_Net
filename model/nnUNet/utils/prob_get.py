import os

import SimpleITK
import nibabel as nib
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import torch
import numpy as np
from scipy.ndimage import label
import argparse


def load_image_file_as_array(*, location, filename):
    # Use SimpleITK to read a file
    result = SimpleITK.ReadImage(os.path.join(location,filename))
    spacing = result.GetSpacing()
    direction = result.GetDirection()
    origin = result.GetOrigin()
    img, props = SimpleITKIO().read_images([os.path.join(location,filename)])
    print(f"Loaded image with shape {img.shape}")
    print('input_files:', filename)
    print('spacing:', spacing)
    print('direction:', direction)
    print('origin:', origin)
    print('props:', props)
    # Convert it to a Numpy array
    return img, spacing, direction, origin, props

def get_prob(input_folder,model,output_folder,mode='single'):
    # Load the image
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_gpu=True,
        device=torch.device('cuda'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    predictor.initialize_from_trained_model_folder(
        model,
        use_folds=(0,1,2,3,4),
        checkpoint_name='checkpoint_final.pth',
    )
    if mode == 'single':
        for filename in os.listdir(input_folder):
            if filename.endswith('.nii.gz'):
                img, spacing, direction, origin, props = load_image_file_as_array(location=input_folder, filename=filename)

                _,prob = predictor.predict_single_npy_array(img, props, None, None, True)
                prob_output_file = os.path.join(output_folder, filename.replace('.nii.gz',''))
                np.savez_compressed(prob_output_file, probabilities=prob,spacing=spacing,direction=direction,origin=origin,props=props)
                print(f"Saved probabilities to {prob_output_file}")
    elif mode == 'batch':
        predictor.predict_from_files_with_prob_only(input_folder,output_folder)

if __name__ == "__main__":

    # get_prob('/home/hcy/nnUNet/DATASET/nnUNet_raw/FirstStage_test_bsb',
    #         '/home/hcy/nnUNet/DATASET/nnUNet_train_models/nntrans/Dataset009_MBHadvance/nnUNetTrainer__nnUNetPlans__2d',
    #         '/home/hcy/nnUNet/DATASET/nnUNet_inference/bsb_nntrans/probabilities_9'
    # )
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_folder', type=str, help='input folder')
    parser.add_argument('--model', type=str, help='model path')
    parser.add_argument('--output_folder', type=str, help='output folder')
    parser.add_argument('--mode', type=str, help='mode',default='batch',required=False)
    args = parser.parse_args()

    get_prob(args.input_folder
             ,args.model,
             args.output_folder,
             args.mode)