import numpy as np
import torch
import torch.nn.functional as F

def cal_confidence(pred_u,num_classes):
    pred_u = F.softmax(pred_u, dim=1)
    # obtain pseudos
    logits_u_aug, _= torch.max(pred_u, dim=1)
    
    # obtain confidence
    entropy = -torch.sum(pred_u * torch.log(pred_u + 1e-10), dim=1)
    entropy /= np.log(num_classes)
    confidence = 1.0 - entropy
    confidence = confidence * logits_u_aug
    confidence = confidence.mean(dim=[1,2,3])  # 1*C
    confidence = confidence.cpu().numpy().tolist()
    # confidence = logits_u_aug.ge(p_threshold).float().mean(dim=[1,2]).cpu().numpy().tolist()
    del pred_u
    return confidence

def rand_bbox3d(size,lam=None):
    if len(size) == 4:
        D = size[1]
        W = size[2]
        H = size[3]
    elif len(size) == 5:
        D = size[2]
        W = size[3]
        H = size[4]
    else:
        raise Exception
    B = size[0]
    cut_rat = np.sqrt(1. - lam)
    cut_d = int(D * cut_rat)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(size=[B, ], low=int(W/8), high=W)
    cy = np.random.randint(size=[B, ], low=int(H/8), high=H)
    ch = np.random.randint(size=[B, ], low=int(D/8), high=D)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbh1 = np.clip(ch - cut_d // 2, 0, D)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    bbh2 = np.clip(ch + cut_d // 2, 0, D)

    return bbh1,bbx1,bby1,bbh2,bbx2,bby2 


def cut_mix_label_adaptive_3d(unlabeled_image, unlabeled_mask, unlabeled_logits, labeled_image, labeled_mask, lst_confidences):
    assert len(lst_confidences) == len(unlabeled_image), "Ensure the confidence is properly obtained"
    assert labeled_image.shape == unlabeled_image.shape, "Ensure shape match between lb and unlb"
    
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    labeled_logits = torch.ones_like(labeled_mask)
    
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]
    
    l_bbh1, l_bbx1, l_bby1, l_bbh2, l_bbx2, l_bby2 = rand_bbox3d(unlabeled_image.size(), lam=np.random.beta(8, 2))
    u_bbh1, u_bbx1, u_bby1, u_bbh2, u_bbx2, u_bby2 = rand_bbox3d(unlabeled_image.size(), lam=np.random.beta(4, 4))
    
    for i in range(mix_unlabeled_image.shape[0]):
        if np.random.random() > lst_confidences[i]:
            mix_unlabeled_image[i, :,l_bbh1[i]:l_bbh2[i], l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = \
                labeled_image[u_rand_index[i],:, l_bbh1[i]:l_bbh2[i], l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]]
            
            mix_unlabeled_target[i, l_bbh1[i]:l_bbh2[i], l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = \
                labeled_mask[u_rand_index[i], l_bbh1[i]:l_bbh2[i], l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]]
            
            mix_unlabeled_logits[i, l_bbh1[i]:l_bbh2[i], l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = \
                labeled_logits[u_rand_index[i], l_bbh1[i]:l_bbh2[i], l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]]
    
    for i in range(unlabeled_image.shape[0]):
        unlabeled_image[i, :,u_bbh1[i]:u_bbh2[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            mix_unlabeled_image[u_rand_index[i],:,u_bbh1[i]:u_bbh2[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        
        unlabeled_mask[i, u_bbh1[i]:u_bbh2[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            mix_unlabeled_target[u_rand_index[i], u_bbh1[i]:u_bbh2[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        
        unlabeled_logits[i, u_bbh1[i]:u_bbh2[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            mix_unlabeled_logits[u_rand_index[i], u_bbh1[i]:u_bbh2[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
    
    del mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits, labeled_logits
    
    return unlabeled_image, unlabeled_mask, unlabeled_logits




# if __name__ == "__main__":
    # lam=np.random.beta(8, 2)
    # print(lam)
    # h1,x1,y1,h2,x2,y2 = rand_bbox3d(size=[1,20,224,224],lam=lam)
    # print(h1,x1,y1,h2,x2,y2)

    # from dataset import SimpleITKPreprocessor,CustomDataset
    # target_size = (224, 224, 20)  # 目标大小
    # normalization_range = (0, 1)  # 归一化范围

    # # 初始化预处理器
    # preprocessor = SimpleITKPreprocessor(target_size=target_size, normalization_range=normalization_range)
    # train_dataset = CustomDataset(root_dir='Dataset006_MBHUns', transform=preprocessor)
    # img,label=train_dataset.__getitem__(20)
    # train_dataset_un = CustomDataset(root_dir='Dataset006_MBHUns',mode='semi',transform=preprocessor)
    # unimg = train_dataset_un.__getitem__(10)
    # # 输入数据的形状
    # input_shape = (1, 20, 224, 224)
    # num_classes = 5

    # # 生成随机概率向量
    # random_probs = np.random.rand(*input_shape, num_classes)

    # # 归一化，使每个像素点的概率向量和为1
    # random_probs = random_probs / random_probs.sum(axis=-1, keepdims=True)

    # # 转换为 PyTorch 张量
    # pred_probs = torch.tensor(random_probs, dtype=torch.float32)

    # # 调整维度顺序为 (1, 5, 20, 224, 224)
    # pred_probs = pred_probs.permute(0, 4, 1, 2, 3)

    # print(pred_probs.shape)  # 应该是 torch.Size([1, 5, 20, 224, 224])

    # pred_u = F.softmax(pred_probs, dim=1)
    #             # obtain pseudos
    # logits_u, label_u = torch.max(pred_u, dim=1)
    # conf = cal_confidence(pred_u,num_classes=5)

    # unlabeled_image, unlabeled_mask, unlabeled_logits= cut_mix_label_adaptive_3d(unimg.unsqueeze(0),label_u,logits_u,img.unsqueeze(0),label.unsqueeze(0),conf)
    
    # print(unlabeled_image.shape,unlabeled_mask.shape,unlabeled_logits.shape)
    # # 创建一个形状为 [1, 1, 20, 224, 224] 的张量示例
    # # 去掉 batch 和 channel 维度，使其形状为 [20, 224, 224]
    # tensor = unlabeled_image.squeeze().numpy()
    # label_tensor = unlabeled_mask.squeeze().numpy().astype(np.int16)
    # import SimpleITK as sitk
    # # 转换为 SimpleITK 图像
    # image = sitk.GetImageFromArray(tensor)
    # label = sitk.GetImageFromArray(label_tensor)
    # # 设置图像方向和空间信息（可选）
    # image.SetSpacing([1.0, 1.0, 1.0])
    # image.SetOrigin([0.0, 0.0, 0.0])
    # image.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    # label.SetSpacing([1.0, 1.0, 1.0])
    # label.SetOrigin([0.0, 0.0, 0.0])
    # label.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

    # # 保存为 NIfTI 文件
    # output_file = 'output_image.nii.gz'
    # output_label = 'output_label.nii.gz'
    # sitk.WriteImage(image, output_file)
    # sitk.WriteImage(label, output_label)

    # print(f'Successfully saved image to {output_file}, label to {output_label}')






