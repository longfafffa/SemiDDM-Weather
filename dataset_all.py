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


class TrainLabeled(data.Dataset):
    def __init__(self, dataroot, phase, finesize):
        super().__init__()
        self.phase = phase
        self.root = dataroot
        self.fineSize = finesize
        # get path
        # input path
        self.dir_snow = os.path.join(self.root, self.phase, 'input', 'snow')
        self.dir_rain = os.path.join(self.root, self.phase, 'input', 'rain')
        self.dir_raindrop = os.path.join(self.root, self.phase, 'input', 'raindrop')

        # gt path
        self.dir_snow_gt = os.path.join(self.root, self.phase, 'gt', 'snow')
        self.dir_rain_gt = os.path.join(self.root, self.phase, 'gt', 'rain')
        self.dir_raindrop_gt = os.path.join(self.root, self.phase, 'gt', 'raindrop')
        # get the dir
        # input dir
        snow_dir = sorted(make_dataset(self.dir_snow))
        rain_dir = sorted(make_dataset(self.dir_rain))
        raindrop_dir = sorted(make_dataset(self.dir_raindrop))

        # gt dir
        snow_dir_gt = sorted(make_dataset(self.dir_snow_gt))
        rain_dir_gt = sorted(make_dataset(self.dir_rain_gt))
        raindrop_dir_gt = sorted(make_dataset(self.dir_raindrop_gt))

        # select snow_dir
        selected_id = random.sample(range(0, len(snow_dir)), 4500)
        select_snow = [snow_dir[i] for i in selected_id]
        select_snow_gt = [snow_dir_gt[i] for i in selected_id]
        self.A_paths = rain_dir + raindrop_dir + select_snow
        self.B_paths = rain_dir_gt + raindrop_dir_gt + select_snow_gt
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

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
        rotate_index = randrange(0, 5)
        rotated_a = rotate(resized_a, rotate_index)
        rotated_b = rotate(resized_b, rotate_index)
        x, y = randrange(width - self.fineSize + 1), randrange(height - self.fineSize + 1)
        degrad_patch = rotated_a.crop((x, y, x + self.fineSize, y + self.fineSize))
        clean_patch = rotated_b.crop((x, y, x + self.fineSize, y + self.fineSize))
        tensor_a = self.transform(degrad_patch)
        tensor_b = self.transform(clean_patch)
        return tensor_a, tensor_b

    def update_paths(self):
        # input path
        self.dir_snow = os.path.join(self.root, self.phase, 'input', 'snow')
        self.dir_rain = os.path.join(self.root, self.phase, 'input', 'rain')
        self.dir_raindrop = os.path.join(self.root, self.phase, 'input', 'raindrop')

        # gt path
        self.dir_snow_gt = os.path.join(self.root, self.phase, 'gt', 'snow')
        self.dir_rain_gt = os.path.join(self.root, self.phase, 'gt', 'rain')
        self.dir_raindrop_gt = os.path.join(self.root, self.phase, 'gt', 'raindrop')
        # get the dir
        # input dir
        snow_dir = sorted(make_dataset(self.dir_snow))
        rain_dir = sorted(make_dataset(self.dir_rain))
        raindrop_dir = sorted(make_dataset(self.dir_raindrop))

        # gt dir
        snow_dir_gt = sorted(make_dataset(self.dir_snow_gt))
        rain_dir_gt = sorted(make_dataset(self.dir_rain_gt))
        raindrop_dir_gt = sorted(make_dataset(self.dir_raindrop_gt))

        # select snow_dir
        selected_id = random.sample(range(len(snow_dir)), 4500)
        select_snow = [snow_dir[i] for i in selected_id]
        select_snow_gt = [snow_dir_gt[i] for i in selected_id]
        self.A_paths = rain_dir + raindrop_dir + select_snow
        self.B_paths = rain_dir_gt + raindrop_dir_gt + select_snow_gt

    def __len__(self):
        return len(self.A_paths)


class TrainUnlabeled(data.Dataset):
    def __init__(self, dataroot, phase, finesize):
        super().__init__()
        self.phase = phase
        self.root = dataroot
        self.fineSize = finesize
        self.dir_snow = os.path.join(self.root, self.phase, 'input', 'snow')
        self.dir_rain = os.path.join(self.root, self.phase, 'input', 'rain')
        self.dir_raindrop = os.path.join(self.root, self.phase, 'input', 'raindrop')
        # self.dir_snow = os.path.join(self.root, self.phase + '/input','snow')
        # gt path
        self.dir_snow_gt = os.path.join(self.root, self.phase, 'candidate', 'snow')
        self.dir_rain_gt = os.path.join(self.root, self.phase, 'candidate', 'rain')
        self.dir_raindrop_gt = os.path.join(self.root, self.phase, 'candidate', 'raindrop')
        # self.dir_D = os.path.join(self.root, self.phase + '/candidate')

        # input dir
        snow_dir = sorted(make_dataset(self.dir_snow))
        rain_dir = sorted(make_dataset(self.dir_rain))
        raindrop_dir = sorted(make_dataset(self.dir_raindrop))

        # gt dir
        snow_dir_gt = sorted(make_dataset(self.dir_snow_gt))
        rain_dir_gt = sorted(make_dataset(self.dir_rain_gt))
        raindrop_dir_gt = sorted(make_dataset(self.dir_raindrop_gt))
        selected_id = random.sample(range(0, len(snow_dir)), 4500)

        select_snow = [snow_dir[i] for i in selected_id]
        select_snow_gt = [snow_dir_gt[i] for i in selected_id]
        self.A_paths = rain_dir + raindrop_dir + select_snow
        self.B_paths = rain_dir_gt + raindrop_dir_gt + select_snow_gt
        # transform
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def __getitem__(self, index):
        A = Image.open(self.A_paths[index]).convert("RGB")
        candidate = Image.open(self.B_paths[index]).convert('RGB')
        resized_a = A
        # resized_b = B
        width, height = A.size
        if width < self.fineSize and height < self.fineSize:
            resized_a = A.resize((self.fineSize, self.fineSize), Image.ANTIALIAS)
            # resized_b = B.resize((self.fineSize, self.fineSize), Image.ANTIALIAS)
            width = self.fineSize
            height = self.fineSize
        elif width < self.fineSize:
            resized_a = A.resize((self.fineSize, height), Image.ANTIALIAS)
            # resized_b = B.resize((self.fineSize, height), Image.ANTIALIAS)
            width = self.fineSize
        elif height < self.fineSize:
            resized_a = A.resize((width, self.fineSize), Image.ANTIALIAS)
            # resized_b = B.resize((width, self.fineSize), Image.ANTIALIAS)
            height = self.fineSize
        # totate
        x, y = (width - self.fineSize + 1) / 2, (height - self.fineSize + 1) / 2
        degrad_patch_crop = resized_a.crop((x, y, x + self.fineSize, y + self.fineSize))
        # strong augmentation
        strong_data = data_aug(degrad_patch_crop)
        tensor_w = self.transform(degrad_patch_crop)
        tensor_s = self.transform(strong_data)
        # tensor_c = self.transform(C)
        tensor_d = self.transform(candidate)
        name = self.B_paths[index]
        return tensor_w, tensor_s, tensor_d, name

    def update_paths(self):
        # input path
        self.dir_snow = os.path.join(self.root, self.phase, 'input', 'snow')
        self.dir_rain = os.path.join(self.root, self.phase, 'input', 'rain')
        self.dir_raindrop = os.path.join(self.root, self.phase, 'input', 'raindrop')

        # gt path
        self.dir_snow_gt = os.path.join(self.root, self.phase, 'candidate', 'snow')
        self.dir_rain_gt = os.path.join(self.root, self.phase, 'candidate', 'rain')
        self.dir_raindrop_gt = os.path.join(self.root, self.phase, 'candidate', 'raindrop')
        # get the dir
        # input dir
        snow_dir = sorted(make_dataset(self.dir_snow))
        rain_dir = sorted(make_dataset(self.dir_rain))
        raindrop_dir = sorted(make_dataset(self.dir_raindrop))

        # gt dir
        snow_dir_gt = sorted(make_dataset(self.dir_snow_gt))
        rain_dir_gt = sorted(make_dataset(self.dir_rain_gt))
        raindrop_dir_gt = sorted(make_dataset(self.dir_raindrop_gt))

        # select snow_dir
        selected_id = random.sample(range(len(snow_dir)), 4500)
        select_snow = [snow_dir[i] for i in selected_id]
        select_snow_gt = [snow_dir_gt[i] for i in selected_id]

        self.A_paths = rain_dir + raindrop_dir + select_snow
        self.B_paths = rain_dir_gt + raindrop_dir_gt + select_snow_gt

    def __len__(self):
        return len(self.A_paths)

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
class Train_undataset(data.Dataset):
    def __init__(self, dataroot, phase, finesize, dataset):
        super().__init__()
        self.phase = phase
        self.root = dataroot
        self.fineSize = finesize
        self.dir = os.path.join(self.root, self.phase, 'input', dataset)
        # gt path
        self.dir_gt = os.path.join(self.root, self.phase, 'candidate', dataset)

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
        candidate = Image.open(self.B_paths[index]).convert('RGB')
        resized_a = A
        # resized_b = B
        width, height = A.size
        if width < self.fineSize and height < self.fineSize:
            resized_a = A.resize((self.fineSize, self.fineSize), Image.ANTIALIAS)
            # resized_b = B.resize((self.fineSize, self.fineSize), Image.ANTIALIAS)
            width = self.fineSize
            height = self.fineSize
        elif width < self.fineSize:
            resized_a = A.resize((self.fineSize, height), Image.ANTIALIAS)
            # resized_b = B.resize((self.fineSize, height), Image.ANTIALIAS)
            width = self.fineSize
        elif height < self.fineSize:
            resized_a = A.resize((width, self.fineSize), Image.ANTIALIAS)
            # resized_b = B.resize((width, self.fineSize), Image.ANTIALIAS)
            height = self.fineSize
        # totate
        x, y = (width - self.fineSize + 1) / 2, (height - self.fineSize + 1) / 2
        degrad_patch_crop = resized_a.crop((x, y, x + self.fineSize, y + self.fineSize))
        # strong augmentation
        strong_data = data_aug(degrad_patch_crop)
        tensor_w = self.transform(degrad_patch_crop)
        tensor_s = self.transform(strong_data)
        # tensor_c = self.transform(C)
        tensor_d = self.transform(candidate)
        name = self.B_paths[index]
        return tensor_w, tensor_s, tensor_d, name

    def update_paths(self):
        self.dir = os.path.join(self.root, self.phase, 'input', 'snow')
        self.dir_gt = os.path.join(self.root, self.phase, 'candidate', 'snow')
        dir = sorted(make_dataset(self.dir))
        dir_gt = sorted(make_dataset(self.dir_gt))
        selected_id = random.sample(range(0, len(dir)), 4500)
        self.A_paths = [dir[i] for i in selected_id]
        self.B_paths = [dir_gt[i] for i in selected_id]

    def __len__(self):
        return len(self.A_paths)
class ValLabeled(data.Dataset):
    def __init__(self, dataroot, finesize):
        super().__init__()
        # self.phase = phase
        self.root = dataroot
        self.fineSize = finesize

        self.dir_A = os.path.join(self.root, 'input')
        self.dir_B = os.path.join(self.root, 'GT')

        # image path
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))

        # transform
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
        elif width < self.fineSize:
            resized_a = A.resize((self.fineSize, height), Image.ANTIALIAS)
            resized_b = B.resize((self.fineSize, height), Image.ANTIALIAS)
        elif height < self.fineSize:
            resized_a = A.resize((width, self.fineSize), Image.ANTIALIAS)
            resized_b = B.resize((width, self.fineSize), Image.ANTIALIAS)
        x, y = (width - self.fineSize + 1) / 2, (height - self.fineSize + 1) / 2
        degrad_patch_crop = resized_a.crop((x, y, x + self.fineSize, y + self.fineSize))
        clean_patch_crop = resized_b.crop((x, y, x + self.fineSize, y + self.fineSize))
        tensor_a = self.transform(degrad_patch_crop)
        tensor_b = self.transform(clean_patch_crop)
        return tensor_a, tensor_b

    def __len__(self):
        return len(self.A_paths)


class TestData(data.Dataset):
    def __init__(self, dataroot):
        super().__init__()
        self.root = dataroot

        self.dir_A = os.path.join(self.root + '/input')
        self.dir_B = os.path.join(self.root + '/GT')
        # self.dir_C = os.path.join(self.root + '/LA')

        # image path
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        # self.C_paths = sorted(make_dataset(self.dir_C))

        # transform
        self.transform = ToTensor()  # [0,1]

    def __getitem__(self, index):
        # A, B is the image pair, hazy, gt respectively
        A = Image.open(self.A_paths[index]).convert("RGB")
        B = Image.open(self.B_paths[index]).convert("RGB")
        # C = Image.open(self.C_paths[index]).convert("RGB")
        # transform to (0, 1)
        tensor_a = self.transform(A)
        tensor_b = self.transform(B)
        # tensor_c = self.transform(C)
        name = self.A_paths[index]
        return tensor_a, tensor_b, name

    def __len__(self):
        return len(self.A_paths)
class TestData_real(data.Dataset):
    def __init__(self, dataroot):
        super().__init__()
        self.root = dataroot

        self.dir_A = self.root
        # image path
        self.A_paths = sorted(make_dataset(self.dir_A))
        # transform
        self.transform = ToTensor()  # [0,1]

    def __getitem__(self, index):
        # A, B is the image pair, hazy, gt respectively
        A = Image.open(self.A_paths[index]).convert("RGB")
        # transform to (0, 1)
        tensor_a = self.transform(A)
        # tensor_b = self.transform(B)
        # tensor_c = self.transform(C)
        name = self.A_paths[index]
        return tensor_a, tensor_a, name

    def __len__(self):
        return len(self.A_paths)


def data_aug(images):
    kernel_size = int(random.random() * 4.95)
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    blurring_image = transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
    color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25)
    strong_aug = images
    if random.random() < 0.8:
        strong_aug = color_jitter(strong_aug)
    strong_aug = transforms.RandomGrayscale(p=0.2)(strong_aug)
    if random.random() < 0.5:
        strong_aug = blurring_image(strong_aug)
    return strong_aug
