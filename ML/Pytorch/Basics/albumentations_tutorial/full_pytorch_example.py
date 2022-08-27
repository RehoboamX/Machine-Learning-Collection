import torch
import numpy as np
import cv2
from PIL import Image
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

class ImageFolder(nn.Module):
    def __init__(self, root_dir, transform=None):   # 把transform函数当作参数传入
        super(ImageFolder, self).__init__()
        self.data = []  # 应为一个形式上为[(cat_0.jpg, 0), ..., (dog_0.jpg, 1), ...]的数组
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = os.listdir(root_dir)  # 返回根目录下子目录的名称的列表作为class_names，这里的例子为['cats', 'dogs']

        for index, name in enumerate(self.class_names):  # 这里index可以直接被用作标签
            files = os.listdir(os.path.join(self.root_dir, name))   # 返回子目录下的文件名组成的列表
            self.data += list(zip(files, [index]*len(files)))   # 列表中为图片名称(str)和标签(int)构成的元组

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file, label = self.data[index]
        root_and_dir = os.path.join(self.root_dir, self.class_names[label])
        image = np.array(Image.open(os.path.join(root_and_dir, img_file)))

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image, label


transforms = A.Compose(
    [
        A.Resize(width=1920, height=1080),
        A.RandomCrop(width=1280, height=720),
        A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),  # 以90%的概率转动40°
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),    # 对RGB每个通道随机移动值
        A.OneOf([
            A.Blur(blur_limit=3, p=0.5),    # 两个操作之一会被随机选择
            A.ColorJitter(p=0.5),   # 改变亮度、对比度、饱和度和色调
        ], p=1.0),   # OneOf被执行的概率为1
        A.Normalize(    # 在torchvision中，要先ToTensor()，再Normalize(mean, stds)
            mean=[0, 0, 0],
            std=[1, 1, 1],
            max_pixel_value=255,
        ),
        ToTensorV2(),   # albumentations.pytorch中的ToTensorV2()是不进行归一化的，需要在之前对数据进行归一化
    ]
)

dataset = ImageFolder(root_dir="cat_dogs", transform=transforms)

for x, y in dataset:
    print(x.shape)
