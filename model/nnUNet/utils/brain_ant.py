import ants

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--fix_path', type=str, required=True)
parser.add_argument('--move_path', type=str, required=True)
parser.add_argument('--move_label_path', type=str, required=True)
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--save_label_path', type=str, required=True)

args = parser.parse_args()
### 将上述的代码封装成函数
def ants_registration(fix_path, move_path, move_label_path, save_path, save_label_path):
    fix_img = ants.image_read(fix_path)
    move_img = ants.image_read(move_path)
    move_label_img = ants.image_read(move_label_path)
    outs = ants.registration(fix_img, move_img, type_of_transforme='SyN')
    reg_img = outs['warpedmovout']
    print(f"writing image to {save_path}",f"writing label to {save_label_path}")
    ants.image_write(reg_img, save_path)
    reg_label_img = ants.apply_transforms(fix_img, move_label_img, transformlist=outs['fwdtransforms'],
                                          interpolator='nearestNeighbor')
    ants.image_write(reg_label_img, save_label_path)

def batch_process(fix_path, move_dir, move_label_dir, save_dir, save_label_dir):
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_label_dir):
        os.makedirs(save_label_dir)
    move_list = [file for file in os.listdir(move_dir) if file.endswith('.nii.gz')]
    # move_label_list = [file for file in os.listdir(move_label_dir) if file.endswith('.nii.gz')]
    for file in move_list:
        move_path = os.path.join(move_dir, file)
        move_label_path = os.path.join(move_label_dir, file.replace('_0000', ''))
        print(f"processing{move_path} and {move_label_path}")
        save_path = os.path.join(save_dir, file)
        save_label_path = os.path.join(save_label_dir, file.replace('_0000', ''))
        ants_registration(fix_path, move_path, move_label_path, save_path, save_label_path)

if __name__ == '__main__':
    batch_process(args.fix_path, args.move_path, args.move_label_path, args.save_path, args.save_label_path)