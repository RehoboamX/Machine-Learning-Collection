"""
Implementation of Yolo Loss Function from the original yolo paper

"""

import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    """
    Calculate the loss for yolo (v1) model
    """
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (in paper and VOC dataset is 20),
        """
        self.S = S
        self.B = B
        self.C = C

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, targets):
        # predictions are shaped (BATCH_SIZE, S, S, (C+B*5)) when inputted
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5)  # -1为bs的维度

        # Calculate IoU for the two predicted bounding boxes with target bbox
        iou_b1 = intersection_over_union(predictions[..., 21:25], targets[..., 21:25])  # ...指代bs，S,S三个维度
        iou_b2 = intersection_over_union(predictions[..., 26:30], targets[..., 21:25])  # 输出tensor的shape：(bs, s, s, 1)
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)  # 将两tensor新增0维，并在0维拼接, 输出:(2, bs, s, s, 1)

        # Take the box with highest IoU out of the two prediction
        # Note that bestbox will be indices of 0, 1 for which bbox was best
        iou_maxes, bestbox = torch.max(ious, dim=0)  # iou_max（bs, s, s, 1）为最大的IOU值; bestbbox（bs, s, s, 1）为索引，由0和1组成
        exists_box = targets[..., 20].unsqueeze(3)  # 输出：(bs, s, s, 1), 20位判断cell中是否有目标（0或1）, unsqueeze(3)可换成20：21（防降维）

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # Set boxes with no object in them to 0. We only take out one of the two
        # predictions, which is the one with highest Iou calculated previously.
        box_predictions = exists_box * (   # 对于没有目标的cells，将其对应的四个坐标值都置0；形状：（bs, s, s, 4）
            (   # 括号里取了两个框中IOU大的框对应的4个坐标，形状（bs, s, s, 4）
                bestbox * predictions[..., 26:30]   # 第二个bbox的IOU最大对应bestbox为1
                + (1 - bestbox) * predictions[..., 21:25]   # 第一个bbox的IOU最大对应bestbox为0
            )
        )

        box_targets = exists_box * targets[..., 21:25]  # 对于targets执行同样的操作

        # Take sqrt of width, height of boxes （见原论文）
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(  # 后面加了绝对值函数，这里要保证符号与之前一致
            torch.abs(box_predictions[..., 2:4] + 1e-6)  # 训练初期初始化时可能为负，要加绝对值；1e-6的作用是防止sqrt(0)得到nan梯度
        )

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])  # ground truth标签的高和宽不可能为负值

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),  # (bn, s, s, 4) -> (bn*s*s, 4)
            torch.flatten(box_targets, end_dim=-2),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = (    # 20：21索引的目的是防止降维, 形状（bs, s, s, 1）
            bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]  # 取每个cell中IOU最大的bbox的置信概率
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),   # 形状：（bs*s*s）
            torch.flatten(exists_box * targets[..., 20:21]),
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        # max_no_obj = torch.max(predictions[..., 20:21], predictions[..., 25:26])
        # no_object_loss = self.mse(
        #    torch.flatten((1 - exists_box) * max_no_obj, start_dim=1),
        #    torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        # )  # 这是原论文中设计的形式，只用cell中IOU最大的bbox计算；在本代码中做了修改，对每个cell中两个bbox都做了计算

        # (bs, s, s, 1) -> (bs, s*s*1)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),  # 选出exist_box=0的predictions框的置信概率
            torch.flatten((1 - exists_box) * targets[..., 20:21], start_dim=1)
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),  # 选出exist_box=0的predictions框的置信概率
            torch.flatten((1 - exists_box) * targets[..., 20:21], start_dim=1)
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        # (bs, s, s, 20) -> (bs*s*s, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),   # 将不存在目标的cell中的类概率置0
            torch.flatten(exists_box * targets[..., :20], end_dim=-2),
        )

        loss = (
            self.lambda_coord * box_loss  # 论文中的前两项损失
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss
