import cv2
import albumentations as A
import numpy as np
from utils import plot_examples
from PIL import Image

image = Image.open("images/elon.jpeg")
mask = Image.open("images/mask.jpeg")
mask2 = Image.open("images/second_mask.jpeg")

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
        ], p=1.0)   # OneOf被执行的概率为1
    ]
)

images_list = [image]
image = np.array(image)  # 将PIL格式的图片转换为numpy array格式
mask = np.array(mask)
mask2 = np.array(mask2)

for i in range(4):
    augementations = transforms(image=image, masks=[mask, mask2])    # 返回一个字典
    augemented_img = augementations["image"]    # key对应的value
    augemented_masks = augementations["masks"]
    images_list.append(augemented_img)
    images_list.append(augemented_masks[0])
    images_list.append(augemented_masks[1])
plot_examples(images_list)