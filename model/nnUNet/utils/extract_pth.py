
import os
import shutil

# 指定的数据集和trainer路径
source_trainer_path = '/home/hcy/nnUNet/DATASET/nnUNet_train_models/nnsam/Dataset009_MBHadvance/nnUNetTrainer__nnUNetPlans__2d'
# 保存提取 checkpoint_final.pth 文件的目标路径
target_dir = '/home/hcy/SecondStage_modify_version/v6/nnsam/Dataset009_MBHadvance'

def extract_checkpoint_final(source_trainer_path, target_dir):
    # 遍历 trainer 路径中的 fold 目录
    for fold_folder in os.listdir(source_trainer_path):
        fold_folder_path = os.path.join(source_trainer_path, fold_folder)

        if os.path.isdir(fold_folder_path):
            # 构建 checkpoint_final.pth 的完整路径
            checkpoint_file = os.path.join(fold_folder_path, 'checkpoint_final.pth')

            # 如果文件存在，复制到目标目录
            if os.path.exists(checkpoint_file):
                # 构建目标目录结构（保持相同的目录层次）
                relative_path = os.path.relpath(fold_folder_path, start=os.path.dirname(source_trainer_path))
                target_checkpoint_folder = os.path.join(target_dir, relative_path)
                os.makedirs(target_checkpoint_folder, exist_ok=True)

                # 复制 checkpoint_final.pth
                target_checkpoint_file = os.path.join(target_checkpoint_folder, 'checkpoint_final.pth')
                shutil.copy(checkpoint_file, target_checkpoint_file)
                print(f"Copied {checkpoint_file} to {target_checkpoint_file}")
            else:
                print(f"checkpoint_final.pth not found in {fold_folder_path}")

# 调用函数
extract_checkpoint_final(source_trainer_path, target_dir)

print("Extraction complete.")


