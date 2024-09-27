import collections.abc
import numpy as np
import scipy.stats as stats
import random
import collections
import SimpleITK as sitk
import torch
import cv2
from torchvision import transforms
from PIL import Image,ImageOps,ImageEnhance,ImageFilter
    
class SitkToNumpy(object):
    def __call__(self, *inputs):
        assert len(inputs) == 1 or len(inputs) == 2
        if len(inputs) == 1:
            image = sitk.GetArrayFromImage(inputs[0])
            return image
        else:
            image = sitk.GetArrayFromImage(inputs[0])
            label = sitk.GetArrayFromImage(inputs[1])
            return image,label

class NumpyToSitk(object):
    def __call__(self, *inputs):
        assert len(inputs) == 1 or len(inputs) == 2
        if len(inputs) == 1:
            image = sitk.GetImageFromArray(inputs[0])
            return image
        else:
            image = sitk.GetImageFromArray(inputs[0])
            label = sitk.GetImageFromArray(inputs[1])
            return image,label

class ReadImage(object):
    def __call__(self, *paths):
        assert len(paths) == 1 or len(paths) == 2
        if len(paths) == 1:
            image = sitk.ReadImage(paths[0])
            imageinfo = [image.GetSpacing(),image.GetOrigin(),image.GetDirection()]
            return image,imageinfo
        else:
            image = sitk.ReadImage(paths[0])
            label = sitk.ReadImage(paths[1])
            imageinfo = [image.GetSpacing(),image.GetOrigin(),image.GetDirection()]
            return image,label,imageinfo

class WriteImage(object):
    def __call__(self, *inputs):
        assert len(inputs) == 3 or len(inputs) == 5
        if len(inputs) == 3:
            image = inputs[0]
            imageinfo = inputs[1]
            image.SetSpacing(imageinfo[0])
            image.SetOrigin(imageinfo[1])
            image.SetDirection(imageinfo[2])
            output_file = inputs[2]

            sitk.WriteImage(image, output_file)
        else:
            image = inputs[0]
            imageinfo = inputs[1]
            label = inputs[2]
            output_file = inputs[3]
            output_label = inputs[4]
            image.SetSpacing(imageinfo[0])
            image.SetOrigin(imageinfo[1])
            image.SetDirection(imageinfo[2])
            sitk.WriteImage(image, output_file)
            sitk.WriteImage(label, output_label)
    
class ToTensorNormalize(object):
    def __init__(self, normalization_range=(0, 1)):
        self.normalization_range = normalization_range
        self.to_tensor = transforms.ToTensor()
    def __call__(self, image,label):
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * (self.normalization_range[1] - self.normalization_range[0]) + self.normalization_range[0]
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int16) 
        return image,label

class SitkSamplerResize(object):
    def __init__(self, base_size, ratio_range,scale=True,bigger_side_to_base=True):
        assert isinstance(ratio_range,collections.abc.Iterable) and len(ratio_range) == 2
        self.base_size = base_size
        self.ratio_range = ratio_range
        self.scale = scale
        self.bigger_side_to_base = bigger_side_to_base
    def __call__(self, image, label):
        size = image.GetSize()
        spacing = image.GetSpacing()
        size = [int(size[i] * spacing[i]) for i in range(3)]
        if self.scale:
            ratio = random.uniform(self.ratio_range[0], self.ratio_range[1])
        else:
            ratio = 1
        if self.bigger_side_to_base:
            scale = max(size) / self.base_size
        else:
            scale = min(size) / self.base_size
        new_size = [int(size[i] / scale * ratio) for i in range(3)]
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(spacing)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetInterpolator(sitk.sitkLinear)
        image = resampler.Execute(image)
        label = resampler.Execute(label)
        return image,label

class Crop(object):
    def __init__(self,crop_size,crop_type='rand',ignore_value=255):
        if (isinstance(crop_size, collections.abc.Iterable) and len(crop_size) == 3):
            self.crop_d,self.crop_h,self.crop_w = crop_size
        else:
            raise ValueError("crop_size should be a list or tuple with 3 elements")
        self.crop_type = crop_type
        self.ignore_value = ignore_value
    def __call__(self, image:np.ndarray, label:np.ndarray):
        size = image.shape
        if self.crop_type == 'rand':
            start_d = random.randint(0, size[0] - self.crop_d)
            start_h = random.randint(0, size[1] - self.crop_h)
            start_w = random.randint(0, size[2] - self.crop_w)
        elif self.crop_type == 'center':
            start_d = (size[0] - self.crop_d) // 2
            start_h = (size[1] - self.crop_h) // 2
            start_w = (size[2] - self.crop_w) // 2
        else:
            raise ValueError("crop_type should be 'rand' or 'center'")
        end_d = start_d + self.crop_d
        end_h = start_h + self.crop_h
        end_w = start_w + self.crop_w
        image = image[start_d:end_d, start_h:end_h, start_w:end_w]
        label = label[start_d:end_d, start_h:end_h, start_w:end_w]
        return image,label
    
class RandomFlip(object):
    def __init__(self, flip_ratio=0.5):
        self.flip_ratio = flip_ratio
    def __call__(self, image, label):
        if random.random() < self.flip_ratio:
            image = np.flip(image, axis=0)
            label = np.flip(label, axis=0)
        if random.random() < self.flip_ratio:
            image = np.flip(image, axis=1)
            label = np.flip(label, axis=1)
        if random.random() < self.flip_ratio:
            image = np.flip(image, axis=2)
            label = np.flip(label, axis=2)
        return image,label

class ThreeDimImageSlicer(object):
    def __init__(self, slice_axis):
        self.slice_axis = slice_axis
        self.slice_image = []
    def list2PILimage(self):
        images = []
        for image in self.slice_image:
            image = Image.fromarray(image)
            images.append(image)
        return images
    
    def __call__(self, image):
        if self.slice_axis == 0:
            for i in range(image.shape[0]):
                self.slice_image.append(image[i])
        elif self.slice_axis == 1:
            for i in range(image.shape[1]):
                self.slice_image.append(image[:,i,:])
        elif self.slice_axis == 2:
            for i in range(image.shape[2]):
                self.slice_image.append(image[:,:,i])
            raise ValueError("slice_axis should be 0, 1 or 2")
        
        return self.list2PILimage() 

class Slicer2ThreeDimImage(object):
    def __init__(self, slice_axis):
        self.slice_axis = slice_axis
    def __call__(self, images):
        images = [np.array(image) for image in images]
        image = np.stack(images, axis=self.slice_axis)
        return image

        
def img_aug_identity(img, scales):
    return img

def img_aug_autocontrast(img, scales):
    if img.mode != 'RGB':
        history_mode = img.mode
        img = img.convert('RGB')
    img = ImageOps.autocontrast(img)
    img = img.convert(history_mode)
    return img

def img_aug_equalize(img, scales):
    if img.mode != 'RGB':
        history_mode = img.mode
        img = img.convert('RGB')
    img = ImageOps.equalize(img)
    img = img.convert(history_mode)
    return img

def img_aug_invert(img, scales):
    if img.mode != 'RGB':
        history_mode = img.mode
        img = img.convert('RGB')
    img = ImageOps.invert(img)
    img = img.convert(history_mode)
    return img

def img_aug_blur(img, scales=[0.1, 2.0]):
    assert scales[0]<scales[1]
    if img.mode != 'RGB':
        history_mode = img.mode
        img = img.convert('RGB')
    sigma = random.uniform(scales[0], scales[1])
    img = img.filter(ImageFilter.GaussianBlur(sigma))
    img = img.convert(history_mode)
    return img

def img_aug_brightness(img,scales=[0.05,0.95]):
    if img.mode != 'RGB':
        history_mode = img.mode
        img = img.convert('RGB')
    minv,maxv = scales[0],scales[1]
    v = float(maxv - minv)*random.random()
    v = maxv - v
    img = ImageEnhance.Brightness(img).enhance(v)
    img = img.convert(history_mode)
    return img

def img_aug_color(img,scales=[0.05,0.95]):
    if img.mode != 'RGB':
        history_mode = img.mode
        img = img.convert('RGB')
    minv,maxv = scales[0],scales[1]
    v = float(maxv - minv)*random.random()
    v = maxv - v
    img = ImageEnhance.Color(img).enhance(v)
    img = img.convert(history_mode)
    return img

def img_aug_sharpness(img,scales=[0.05,0.95]):

    if img.mode != 'RGB':
        history_mode = img.mode
        img = img.convert('RGB')
    minv,maxv = scales[0],scales[1]
    v = float(maxv - minv)*random.random()
    v = maxv - v
    img = ImageEnhance.Sharpness(img).enhance(v)
    img = img.convert(history_mode)
    return img

def img_aug_posterize(img,scales=[4,8]):
    if img.mode != 'RGB':
        history_mode = img.mode
        img = img.convert('RGB')
    min_v,max_v = min(scales),max(scales)
    v = float(max_v - min_v)*random.random()
    v = int(np.ceil(v))
    v = max(1,v)
    v = max_v - v
    img = ImageOps.posterize(img,v)
    img = img.convert(history_mode)
    return img

def img_aug_solarize(img,scales=[0.05,0.95]):
    if img.mode != 'RGB':
        history_mode = img.mode
        img = img.convert('RGB')
    minv,maxv = scales[0],scales[1]
    v = float(maxv - minv)*random.random()
    v = maxv - v
    img = ImageOps.solarize(img,v)
    img = img.convert(history_mode)
    return img

def get_augment_list():
    return  [
            (img_aug_identity, None),
            (img_aug_autocontrast, None),
            (img_aug_equalize, None),
            (img_aug_blur, [0.1, 2.0]),
            (img_aug_brightness, [0.05, 0.95]),
            (img_aug_color, [0.05, 0.95]),
            (img_aug_sharpness, [0.05, 0.95]),
            (img_aug_posterize, [4, 8]),
            (img_aug_solarize, [1, 256]),
        ]
class strong_img_aug:
    def __init__(self, num_augs, flag_using_random_num=False):
        assert 1<= num_augs <= 11
        self.n = num_augs
        self.augment_list = get_augment_list()
        self.flag_using_random_num = flag_using_random_num
    
    def __call__(self, slicers):
        if self.flag_using_random_num:
            max_num = np.random.randint(1, high=self.n + 1)
        else:
            max_num =self.n
        ops = random.choices(self.augment_list, k=max_num)
        for op, scales in ops:
            # print("="*20, str(op))
            slicers = [op(slicer, scales) for slicer in slicers]
        return slicers