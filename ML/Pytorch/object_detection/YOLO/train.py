"""
Main file for training Yolo model on Pascal VOC dataset

"""

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import (
    intersection_over_union,
    non_max_suppression,
    mean_average_precision,
    cellboxes_to_boxes,  # 将相对于cell的bboxes转化为相对于整个image的bboxes
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss

seed = 123
torch.manual_seed(seed)  # 设置 CPU 生成随机数的 种子

# Hyperparameters etc.
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
WEIGHT_DECAY = 0  # 设置为0简化运算
EPOCHS = 100
NUM_WORKERS = 1
PIN_MEMORY = True  # 计算机内存充足时或者load的dataset很小时设置
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes  # 因为这里的数据增强只有Resize操作，因此不需改变标签

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])  # PIL转为Tensor：W x H  -->  C x H x W


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the progress bar
        loop.set_postfix(loss=loss.item())  # 对每个batch，进度条更新时打印当前的loss

    print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")  # 打印该epoch下每个batch的平均损失


def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    if LOAD_MODEL:  # 如果有保存的模型
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset(
        "data/8examples.csv",  # 8examples是训练集中的8个样本，先在此训练确定模型正常工作
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        transform=transform,
    )

    test_dataset = VOCDataset(
        "data/test.csv", img_dir=IMG_DIR, label_dir=LABEL_DIR, transform=transform,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,  # 生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转移到GPU的显存就会更快一些
        shuffle=True,
        drop_last=False,  # 当最后的batch只有1-2个数据时更新梯度时会影响训练结果，将其舍去
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    for epoch in range(EPOCHS):
#         for x, y in train_loader:
#             x = x.to(DEVICE)
#             for idx in range(8):
#                 bboxes = cellboxes_to_boxes(model(x))  # 将相对于cell的bboxes转化为相对于整个image的bboxes [BS x S*S x 6]
#                 bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
#                 plot_image(x[idx].permute(1, 2, 0).to("cpu"), bboxes)  # 要将tensor转换为ndarray的形式（只有cpu才能和ndarray互相转换）

#             import sys
#             sys.exit()  # 可视化后退出

        pred_boxes, target_boxes = get_bboxes(  # 返回经过NMS的列表[[train_idx, class_pred, prob_score, x1, y1, x2, y2], ...]
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )

        print(f"Train mAP: {mean_avg_prec}")  # 每轮训练前先计算mAP并打印

#         if mean_avg_prec > 0.9:
#             checkpoint = {
#                 "state_dict": model.state_dict(),
#                 "optimizer": optimizer.state_dict(),
#             }
#             save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
#             import time
#             time.sleep(10)  # 防止重复保存，有时间退出

        train_fn(train_loader, model, optimizer, loss_fn)


if __name__ == "__main__":
    main()
