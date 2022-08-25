# Imports
import os
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from custom_dataset import CatsAndDogsDataset

# Load data
my_transforms = transforms.Compose([
    transforms.ToPILImage(),    # 将图像转换为PILImage对象,很多数据增强操作都要实施在这个对象上
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomVerticalFlip(p=0.05),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])  # (value - mean) / std  (Note: this does nothing!)
])

dataset = CatsAndDogsDataset(csv_file='cats_dogs.csv', root_dir='cats_dogs_resized',
                             transform=my_transforms)

img_num = 0

save_path = "./dataset_transforms"
os.makedirs(save_path, exist_ok=True)

for _ in range(5):
    for img, label in dataset:
        save_image(img, save_path + '\\' + 'img'+str(img_num)+'.png')
        img_num += 1