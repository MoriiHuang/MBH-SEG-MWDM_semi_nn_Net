import os
from os import listdir, makedirs
from os.path import join, isfile, basename
from glob import glob
from tqdm import tqdm
from copy import deepcopy
from time import time
import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT
from matplotlib import pyplot as plt
import cv2
import torch.multiprocessing as mp
import nibabel as nib

import argparse
parser = argparse.ArgumentParser()

parser.add_argument(
    '-pred_save_dir',
    type=str,
    required=True,
    help='directory to save the prediction',
)


parser.add_argument(
    '--ref_nii_file_path',
    type=str,
    default=None,
    help='whether to overwrite the existing prediction'
)

if __name__ == '__main__':
    args = parser.parse_args()
    pred_save_dir = args.pred_save_dir
    ref_nii_file_path = args.ref_nii_file_path

    if ref_nii_file_path is not None:
        ref_nii_files = [f for f in listdir(ref_nii_file_path) if f.endswith('.nii.gz')]
        pred_nii_files = [ f for f in listdir(pred_save_dir) if f.endswith('.nii.gz')]
        for ref_nii_file in ref_nii_files:
            if ref_nii_file not in pred_nii_files:
                print(f'cp {join(ref_nii_file_path,ref_nii_file)} {pred_save_dir}')
                os.system(f'cp {join(ref_nii_file_path,ref_nii_file)} {pred_save_dir}')
