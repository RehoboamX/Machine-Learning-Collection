import torch
from iou import intersection_over_union


def nms(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Does Non Max Suppression given bboxes
    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes
    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    # bboxes = [[1, 0.9, x1, y1, x2, y2], [], []]  # 每个bboxes应该传入6个参数：类别，概率，4个坐标
    assert type(bboxes) == list  # 只有传入的bboxes为列表时程序才能继续执行

    bboxes = [box for box in bbboxes if box[1] > prob_threshold]  # 只考虑概率大于prob_threshold的bboxes
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)   # 先将bboxes按照概率由大到小排列
    bboxes_after_nms = []

    while bboxes:   # 当bboxes列表中还有元素时
        chosen_box = bboxes.pop(0)  # 将概率最大的bbox弹出作为chosen_box与其他的bboxes比较

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]  # 不保留与chosen_box类别不同的bboxes
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),  # 只取出关于坐标的四个元素
                torch.tensor(box[2:]),
                box_format=box_format,
                )
            < iou_threshold  # 当同一类别的bboxes之间的IOU小于iou_threshold时我们才当作其不是在预测同一个对象
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms