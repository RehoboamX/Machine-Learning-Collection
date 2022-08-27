import cv2
import albumentations as A
import numpy as np
from utils import plot_examples
from PIL import Image

image = cv2.imread("images/cat.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2读入的图片形式为BGR的，一般需要转为RGB格式进行使用
boxes = [[13, 170, 224, 410]]  # 这里的bbox是按照两个corner的坐标来定义的

# Pascal_voc (x_min, y_min, x_max, y_max), YOLO和COCO等定义的bbox都不太一样

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
    ], bbox_params=A.BboxParams(format="pascal_voc", min_area=2048,   # 有时数据增强后被检测的对象没有被保留在图像中，这里的min_area指的是bboxes的部分增强后至少有2048个像素
                                min_visibility=0.3, label_fields=[])    # 这里的min_visibility指的是bboxes的部分增强后至少占总图像的30%
)

images_list = [image]   # 之前用PIL读入图片时需要转换为np array，这里用opencv读入不需要
saved_bboxes = [boxes[0]]  # 这里的saved_bboxes与之前定义的bboxes没有区别

for i in range(15):
    augementations = transforms(image=image, bboxes=boxes)    # 这里是把image和bboxes都拿来作数据增强 （注意！ 这里image和bboxes的键名是固定的，不要改)
    augemented_img = augementations["image"]    # key对应的value

    if len(augementations["bboxes"]) == 0:  # 加了如min_area和min_visibility等限定条件后，会drop掉一部分不合适的bboxes
        continue

    images_list.append(augemented_img)
    saved_bboxes.append(augementations["bboxes"][0])  # augementations["bboxes"]返回的是一个列表

plot_examples(images_list, saved_bboxes)