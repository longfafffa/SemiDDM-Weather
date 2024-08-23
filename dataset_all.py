import os.path
import torch
import torch.utils.data as data
from PIL import Image
import random
from random import randrange
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def crop_img(image, base=64):
    h = image.shape[0]
    w = image.shape[1]
    crop_h = h % base
    crop_w = w % base
    return image[crop_h // 2:h - crop_h + crop_h // 2, crop_w // 2:w - crop_w + crop_w // 2, :]


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def rotate(img, rotate_index):
    '''
    :return: 8 version of rotating image
    '''
    if rotate_index == 0:
        return img
    if rotate_index == 1:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    if rotate_index == 2:
        return img.transpose(Image.ROTATE_90)
    if rotate_index == 3:
        return img.transpose(Image.ROTATE_180)
    if rotate_index == 4:
        return img.transpose(Image.ROTATE_270)

class Train_dataset(data.Dataset):
    def __init__(self, dataroot, phase, finesize, dataset):
        super().__init__()
        self.phase = phase
        self.root = dataroot
        self.fineSize = finesize
        self.dir = os.path.join(self.root, self.phase, 'input', dataset)
        # gt path
        self.dir_gt = os.path.join(self.root, self.phase, 'gt', dataset)

        # get the dir
        # input dir
        dir = sorted(make_dataset(self.dir))
        dir_gt = sorted(make_dataset(self.dir_gt))
        # select snow_dir
        if dataset == 'snow':
            selected_id = random.sample(range(0, len(dir)), 4500)
            self.A_paths = [dir[i] for i in selected_id]
            self.B_paths = [dir_gt[i] for i in selected_id]
        else:
            self.A_paths = dir
            self.B_paths = dir_gt
        self.transform = ToTensor()  # [0,1]

    def __getitem__(self, index):
        A = Image.open(self.A_paths[index]).convert("RGB")
        B = Image.open(self.B_paths[index]).convert("RGB")
        resized_a = A
        resized_b = B
        width, height = A.size
        if width < self.fineSize and height < self.fineSize:
            resized_a = A.resize((self.fineSize, self.fineSize), Image.ANTIALIAS)
            resized_b = B.resize((self.fineSize, self.fineSize), Image.ANTIALIAS)
            width = self.fineSize
            height = self.fineSize
        elif width < self.fineSize:
            resized_a = A.resize((self.fineSize, height), Image.ANTIALIAS)
            resized_b = B.resize((self.fineSize, height), Image.ANTIALIAS)
            width = self.fineSize
        elif height < self.fineSize:
            resized_a = A.resize((width, self.fineSize), Image.ANTIALIAS)
            resized_b = B.resize((width, self.fineSize), Image.ANTIALIAS)
            height = self.fineSize
        # rotate
        rotate_index = randrange(0, 2)
        rotated_a = rotate(resized_a, rotate_index)
        rotated_b = rotate(resized_b, rotate_index)
        x, y = randrange(width - self.fineSize + 1), randrange(height - self.fineSize + 1)
        degrad_patch = rotated_a.crop((x, y, x + self.fineSize, y + self.fineSize))
        clean_patch = rotated_b.crop((x, y, x + self.fineSize, y + self.fineSize))
        tensor_a = self.transform(degrad_patch)
        tensor_b = self.transform(clean_patch)
        return tensor_a, tensor_b

    def update_paths(self):
        self.dir = os.path.join(self.root, self.phase, 'input', 'snow')
        self.dir_gt = os.path.join(self.root, self.phase, 'gt', 'snow')
        dir = sorted(make_dataset(self.dir))
        dir_gt = sorted(make_dataset(self.dir_gt))
        selected_id = random.sample(range(0, len(dir)), 4500)
        self.A_paths = [dir[i] for i in selected_id]
        self.B_paths = [dir_gt[i] for i in selected_id]

    def __len__(self):
        return len(self.A_paths)

class Test_dataset(data.Dataset):
    def __init__(self, dataroot):
        super().__init__()
        self.root = dataroot
        self.dir_A = self.root
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.transform = ToTensor()  # [0,1]

    def __getitem__(self, index):
        A = Image.open(self.A_paths[index]).convert("RGB")
        tensor_a = self.transform(A)
        name = self.A_paths[index]
        return tensor_a, tensor_a, name

    def __len__(self):
        return len(self.A_paths)
    
    
class Val_dataset(data.Dataset):
    def __init__(self, dataroot, finesize):
        super().__init__()
        self.root = dataroot
        self.fineSize = finesize
        self.dir_A = dataroot
        self.A_paths = sorted(make_dataset(self.dir_A))
        # transform
        self.transform = ToTensor()  # [0,1]

    def __getitem__(self, index):
        image = Image.open(self.A_paths[index]).convert("RGB")
        resized_a = image
        width, height = image.size
        if width < self.fineSize and height < self.fineSize:
            resized_a = image.resize((self.fineSize, self.fineSize), Image.ANTIALIAS)

        elif width < self.fineSize:
            resized_a = image.resize((self.fineSize, height), Image.ANTIALIAS)

        elif height < self.fineSize:
            resized_a = image.resize((width, self.fineSize), Image.ANTIALIAS)

        x, y = (width - self.fineSize + 1) / 2, (height - self.fineSize + 1) / 2
        degrad_patch_crop = resized_a.crop((x, y, x + self.fineSize, y + self.fineSize))
        tensor_a = self.transform(degrad_patch_crop)
        path=self.A_paths[index].replace('input','candidate')
        return tensor_a, path

    def __len__(self):
        return len(self.A_paths)


