#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os

"""
PLEASE READ paths.md FOR INFORMATION TO HOW TO SET THIS UP
"""

nnUNet_raw ="/opt/app/model/nnUNet/DATASET/nnUNet_raw"
nnUNet_preprocessed ="/opt/app/model/nnUNet/DATASET/nnUNet_preprocessed"
# MODEL_NAME="nnsam"
# MODEL_NAME = 'nnswin'
MODEL_NAME = os.environ.get('MODEL_NAME')
# MODEL_NAME = 'nntrans'

# teacher_model_weights = "/home/hcy/nnUNet/DATASET/nnUNet_train_models/Dataset003_MBH/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
teacher_model_weights = "/home/hcy/nnUNet/DATASET/nnUNet_train_models/nntrans/Dataset003_MBH/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
if MODEL_NAME is not None:
    nnUNet_results = os.path.join("/opt/app/model/nnUNet/DATASET/nnUNet_train_models", MODEL_NAME)
    if not os.path.isdir(nnUNet_results):
        os.makedirs(nnUNet_results)
else:
    nnUNet_results = "/opt/app/model/nnUNet/DATASET/nnUNet_train_models"

if nnUNet_raw is None:
    print("nnUNet_raw is not defined and nnU-Net can only be used on data for which preprocessed files "
          "are already present on your system. nnU-Net cannot be used for experiment planning and preprocessing like "
          "this. If this is not intended, please read documentation/setting_up_paths.md for information on how to set "
          "this up properly.")

if nnUNet_preprocessed is None:
    print("nnUNet_preprocessed is not defined and nnU-Net can not be used for preprocessing "
          "or training. If this is not intended, please read documentation/setting_up_paths.md for information on how "
          "to set this up.")

if nnUNet_results is None:
    print("nnUNet_results is not defined and nnU-Net cannot be used for training or "
          "inference. If this is not intended behavior, please read documentation/setting_up_paths.md for information "
          "on how to set this up.")
