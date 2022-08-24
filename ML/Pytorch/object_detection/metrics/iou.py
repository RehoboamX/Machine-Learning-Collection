import torch

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):  #
    """
    Calculates intersection over union

    :param boxes_preds: Predictions of Bounding Boxes (BATCH SIZE, 4), tensor
    :param boxes_labels: Correct labels of Bounding Boxes (BATCH SIZE, 4), tensor
    :param box_format: midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2), str

    :return: Intersection over union for all examples, tensor
    """
    if box_format == "midpoint":  # 四个值分别为中心点的坐标，宽和高
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":  # 四个值分别为左上角和右下角的坐标
        box1_x1 = boxes_preds[..., 0:1]  # ...表示之前的所有dims
        box1_y1 = boxes_preds[..., 1:2]  # 1:2前闭后开
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]   # 事实上，boxes_preds的最后一个维度并没有4这个索引，之所以这样做是防止tensor被降维
        box2_x1 = boxes_labels[..., 0:1]  # 如果直接传入3而不是3:4，传入的结果形状就会变为N而不是(N, 1)
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    # 得到两个bounding box的intersection（交集）
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0)的作用是计算当两个bounding box不相交时的情况
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)  # x.clamp()的第一个参数为min，x与之相比如果小于min就将min赋值给x

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))  # 严谨起见加上绝对值
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)    # +1e-6是为了数值运算时的稳定性（防止分母为0的情况）