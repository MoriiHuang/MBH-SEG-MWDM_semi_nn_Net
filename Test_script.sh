
export PATH=$PATH:/home/user/.local/bin
### 1.window the raw input

mkdir -p model/nnUNet/DATASET/nnUNet_nyul/SecondStage_test
mkdir -p model/nnUNet/DATASET/nnUNet_raw/SecondStage_test
python model/nnUNet/utils/window.py --input_folder input --output_folder model/nnUNet/DATASET/nnUNet_nyul/SecondStage_test
nyul-normalize model/nnUNet/DATASET/nnUNet_nyul/SecondStage_test -o model/nnUNet/DATASET/nnUNet_raw/SecondStage_test -mo other -lsh  model/nnUNet/DATASET/nnUNet_raw/normal/standard_histogram.npy --plot-histogram
python model/nnUNet/utils/reback.py --input model/nnUNet/DATASET/nnUNet_nyul/SecondStage_test --output model/nnUNet/DATASET/nnUNet_raw/SecondStage_test

mkdir -p model/nnUNet/DATASET/nnUNet_nyul/SecondStage_test_advanced
mkdir -p model/nnUNet/DATASET/nnUNet_raw/SecondStage_test_advanced
python model/nnUNet/utils/advance_window.py --input_folder input --output_folder model/nnUNet/DATASET/nnUNet_nyul/SecondStage_test_advanced
nyul-normalize model/nnUNet/DATASET/nnUNet_nyul/SecondStage_test_advanced -o model/nnUNet/DATASET/nnUNet_raw/SecondStage_test_advanced -mo other -lsh model/nnUNet/DATASET/nnUNet_raw/advance/standard_histogram.npy --plot-histogram
python model/nnUNet/utils/reback.py --input model/nnUNet/DATASET/nnUNet_nyul/SecondStage_test_advanced --output model/nnUNet/DATASET/nnUNet_raw/SecondStage_test_advanced

### copy input into model/MedSAM/data
cp -r input model/MedSAM/data/input_MBH/imagesTs
### 2. predict the probability map
unset MODEL_NAME

python model/nnUNet/utils/prob_get.py --input_folder model/nnUNet/DATASET/nnUNet_raw/SecondStage_test --model model/nnUNet/DATASET/nnUNet_train_models/Dataset006_MBHUns/nnUNetTrainer__nnUNetPlans__2d --output_folder model/nnUNet/DATASET/nnUNet_inference/AugSeg/probabilities_6 --mode 'single'

export MODEL_NAME=nnsam

python model/nnUNet/utils/prob_get.py --input_folder model/nnUNet/DATASET/nnUNet_raw/SecondStage_test_advanced --model model/nnUNet/DATASET/nnUNet_train_models/nnsam/Dataset009_MBHadvance/nnUNetTrainer__nnUNetPlans__2d --output_folder model/nnUNet/DATASET/nnUNet_inference/nnsam_bsb/probabilities_9 --mode 'single'

python model/nnUNet/utils/prob_get.py --input_folder model/nnUNet/DATASET/nnUNet_raw/SecondStage_test --model model/nnUNet/DATASET/nnUNet_train_models/nnsam/Dataset008_MBHMerge/nnUNetTrainer__nnUNetPlans__2d --output_folder model/nnUNet/DATASET/nnUNet_inference/nnsam_all/probabilities_8 --mode 'single'

export MODEL_NAME=nntrans

python model/nnUNet/utils/prob_get.py --input_folder model/nnUNet/DATASET/nnUNet_raw/SecondStage_test --model model/nnUNet/DATASET/nnUNet_train_models/nntrans/Dataset003_MBH/nnUNetTrainer__nnUNetPlans__2d --output_folder model/nnUNet/DATASET/nnUNet_inference/nntrans/probabilities_3 --mode 'single'

unset MODEL_NAME

### 3. probability map merge to segmentation

python model/nnUNet/utils/prob_conbine.py --overall_mask_folder model/nnUNet/DATASET/nnUNet_inference/nnsam_all/seg --subclass_mask_folder model/nnUNet/DATASET/nnUNet_inference/AugSeg/seg --overall_prob_folder model/nnUNet/DATASET/nnUNet_inference/nnsam_all/probabilities_8 --subclass_prob_folder model/nnUNet/DATASET/nnUNet_inference/AugSeg/probabilities_6 --output_folder model/nnUNet/DATASET/nnUNet_inference/merge/AugSeg_nnSAM
python model/nnUNet/utils/prob_merge.py --input_folders 'model/nnUNet/DATASET/nnUNet_inference/merge/AugSeg_nnSAM model/nnUNet/DATASET/nnUNet_inference/nntrans/probabilities_3 model/nnUNet/DATASET/nnUNet_inference/nnsam_bsb/probabilities_9' --output_folder /opt/app/output_nnunet

### 4. Apply Med SAM for adjustment

mkdir -p model/MedSAM/data/input_MBH/preprocessed/npz

python model/MedSAM/pre_CT_MR.py -img_path model/MedSAM/data/input_MBH/imagesTs/input -img_name_suffix .nii.gz -gt_path /opt/app/output_nnunet  -gt_name_suffix .nii.gz -output_path model/MedSAM/data/input_MBH/preprocessed/npz -num_workers 1 -modality CT -anatomy Abd -window_level 80 -window_width 200 --save_nii --split_ratio 0

mkdir -p /opt/app/output_medsam

python model/MedSAM/inference_3D.py -data_root model/MedSAM/data/input_MBH/preprocessed/npz/MedSAM_test/CT_Abd -pred_save_dir /opt/app/output_medsam -medsam_lite_checkpoint_path model/MedSAM/work_dir/medsam_lite_best.pth  -num_workers 1  --save_mode nii --ref_nii_file_path /opt/app/output_nnunet

mkdir -p /opt/app/output_adjust

### 将/opt/app/output_medsam 全部移动到 /opt/app/output_adjust
mv /opt/app/output_medsam/* /opt/app/output_adjust

python model/MedSAM/inference_diff.py -pred_save_dir /opt/app/output_adjust --ref_nii_file_path /opt/app/output_nnunet

python model/nnUNet/utils/select_mask.py --model1_folder /opt/app/output_adjust --model2_folder /opt/app/output_nnunet  --output_folder /opt/app/output