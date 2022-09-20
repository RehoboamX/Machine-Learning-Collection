"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class VOCDataset(Dataset):
    def __init__(
            self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)  # DataFrame类型的annotations形状：[16551 rows x 2 columns]
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)  # 对于train.csv为16651

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])  # 第index行第2列为第index个文件的标签名
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():  # 以行为单位读，f.readlines()为一个一次性工作的列表
                class_label, x, y, width, height = [  # 这里的x, y, width, height分别指四个坐标在整张图片上归一化后的结果
                    float(x) if float(x) != int(float(x)) else int(x)  # 判断字符串中数据是整型还是浮点型并返回相应类型数据
                    for x in label.replace("\n", "").split()  # 去掉换行符，按空格拆成包含多个字符串的列表
                ]

                boxes.append([class_label, x, y, width, height])  # 将labels中第index个文件转化为相应的整型或浮点型并储存到boxes列表中

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])  # 第index行第1列为第index个文件的图片名
        image = Image.open(img_path)  # PIL打开图片文件  PIL的Size： WxH
        boxes = torch.tensor(boxes)  # 将boxes转化为张量

        if self.transform:
            # image = self.transform(image)
            image, boxes = self.transform(image, boxes)  # 对于目标检测任务来说，在对图片数据增强的同时，也要将bbox坐标进行同样变换

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))  # 最后的维度虽然定义了30个，但实际上只用到25个，定义为30只是为了后面的运算方便
        for box in boxes:
            class_label, x, y, width, height = box.tolist()  # 从张量再转化回列表传到5个变量中
            class_label = int(class_label)  # 确保class_label为整型变量

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)  # 判断标签中的bbox是属于哪个cell的
            x_cell, y_cell = self.S * x - j, self.S * y - i  # 这里求的是相对cell归一化的x和y值

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:

            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)

            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (  # 相对cell的width和height
                width * self.S,
                height * self.S
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, 20] == 0:
                # Set that there exists an object
                label_matrix[i, j, 20] = 1  # 将标签相应的cell中有检测对象置信概率项置1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates  # 将上面求出的相对cell的坐标赋值到label_matrix的相应位置

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1  # 将相应cell的对象类别置信概率置1

        return image, label_matrix

