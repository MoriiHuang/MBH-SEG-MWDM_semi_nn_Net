import os

import SimpleITK
import nibabel as nib
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import torch
import numpy as np
from scipy.ndimage import label
import argparse

parser = argparse.ArgumentParser(description=' models prob merge')

parser.add_argument('--input_folders', type=str, help='mask folders')
parser.add_argument('--output_folder', type=str, help='output folder')
parser.add_argument('--merge_ratio', type=str, help='merge ratio',default=None,required=False)


def write_array_as_image_file(*, location, array, spacing, origin, direction, filename):

    # You may need to change the suffix to .tiff to match the expected output

    image = SimpleITK.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetDirection(direction) # My line
    image.SetOrigin(origin)
    image = SimpleITK.Cast(image, SimpleITK.sitkUInt8)
    #保存成Uint8类型
    SimpleITK.WriteImage(
        image,
        os.path.join(location, filename),
        useCompression=True,
    )

def merge_prob(probs,weight):
    assert len(probs) == len(weight)

    sample = np.load(probs[0],allow_pickle=True)
    spacing = sample['spacing']
    direction = sample['direction']
    origin = sample['origin']
    props = sample['props']
    average = sample['probabilities']
    average = average * weight[0]
    for i in range(1,len(probs)):
        sample = np.load(probs[i],allow_pickle=True)
        average += sample['probabilities'] * weight[i]
    
    return average,spacing,direction,origin,props

def batch_process(prob_folders,output_folder,merge_ratio=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(prob_folders[0]):
        if filename.endswith('.npz'):
            probs = [os.path.join(folder,filename) for folder in prob_folders]
            if merge_ratio is not None:
                merge_ratio = [float(ratio) for ratio in merge_ratio]
                assert len(merge_ratio) == len(probs)
                average,spacing,direction,origin,props = merge_prob(probs,merge_ratio)
            else:
                average,spacing,direction,origin,props = merge_prob(probs,[1/len(probs)]*len(probs))
            ret = average.argmax(0)
            del average
            save_filename = filename.replace('_0000.npz','.nii.gz')
            write_array_as_image_file(location=output_folder,array=ret,spacing=spacing,direction=direction,origin=origin,filename=save_filename)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.merge_ratio is not None:
        batch_process(args.input_folders.split(),args.output_folder,args.merge_ratio.split())
    else:
        batch_process(args.input_folders.split(),args.output_folder)
