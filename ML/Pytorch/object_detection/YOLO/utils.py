import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
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

    if box_format == "corners":  # 四个值分别为左上角和右下角的坐标
        box1_x1 = boxes_preds[..., 0:1]  # ...表示之前的所有dims (BS, S, S)
        box1_y1 = boxes_preds[..., 1:2]  # 1:2前闭后开
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # 事实上，boxes_preds的最后一个维度并没有4这个索引，之所以这样做是防止tensor被降维
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

    return intersection / (box1_area + box2_area - intersection + 1e-6)  # +1e-6是为了数值运算时的稳定性（防止分母为0的情况）


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
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

    # bboxes = [[1, 0.9, x1, y1, x2, y2], [...], [...]]  # 每个bboxes应该传入6个参数：类别，概率，4个坐标
    assert type(bboxes) == list  # 只有传入的bboxes为列表时程序才能继续执行

    bboxes = [box for box in bboxes if box[1] > threshold]  # 只考虑概率大于prob_threshold的bboxes
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)  # 先将bboxes按照概率由大到小排列
    bboxes_after_nms = []

    while bboxes:  # 当bboxes列表中还有元素时
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


def mean_average_precision(
        pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Calculates mean average precision 

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # pred_boxes (list): [[train_idx, class_pred, prob_score, x1, y1, x2, y2], ...]，train_idx是图片编号
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6  # epsilon的定义目的是保持数值稳定性

    for c in range(num_classes):  # 对每个类别进行遍历
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)  # 将所有c类别的预测bboxes放入detections列表

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)  # 将所有c类别的标签bboxes放入ground_truths列表

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])  # counter括号中的部分就是单独把train_idx拿出来的数组

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():  # 遍历python字典键值对时要加.items()
            amount_bboxes[key] = torch.zeros(val)  # amount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)  # 将detections中的元素按照prob_score降序排列
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]  # 找出与该detection框train_idx相同的
                # ground_truths中的元素放入ground_truth_img列表中
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(  # 计算该detection框和同一图片中所有ground_truth_img框(同类别)的IOU
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx  # 找到该detection与之最大IOU的ground_truth在ground_truth_img列表中的序号

            if best_iou > iou_threshold:  # 证明这个prediction框是正确的
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:  # 检查之前是否有cover过这个ground_truth的bbox
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1  # 将TP列表中该样本的位置置1
                    amount_bboxes[detection[0]][best_gt_idx] = 1  # 标记该ground_truth的bbox已经被cover
                else:  # 若找到的ground_truth的bbox已经被cover过
                    FP[detection_idx] = 1  # 将FP列表中该样本的位置置1

            # if IOU is lower than the detection is a false positive
            else:  # IOU没有高于设置的阈值
                FP[detection_idx] = 1

        # [1, 1, 0, 1, 0] -> [1, 2, 2, 3, 3]
        TP_cumsum = torch.cumsum(TP, dim=0)  # dim必须设置
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)  # total_true_bboxes = len(ground_truths)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))  # torch.devide()是元素级别的除法
        precisions = torch.cat((torch.tensor([1]), precisions))  # 在precision首端添加元素1，因为recall-precisions坐标系的第一个点要为（0,1）
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))  # torch.trapz(y, x)计算x-y坐标系下坐标点与x轴围成的面积大小

    return sum(average_precisions) / len(average_precisions)  # 不同类别AP结果的均值


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)  # 将图片转为np array的形式 [height, width, channels]
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)  # 在Figure对象中可以包含一个或者多个Axes对象,每个Axes对象都是一个拥有自己坐标系统的绘图区域。1代表排在一行。
    # Display the image
    ax.imshow(im)  # plt.imshow()传入的图片要为（H x W x C）的形状，因此在从tensor转化时要改一下维度的顺序

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        box = box[2:]  # 前两维分别为类别和类概率
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2  # upper_left_x为相对与图像宽度归一化后的坐标，在0和1之间
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),  # 锚点对应的坐标值的tuple
            box[2] * width,  # bbox实际的宽
            box[3] * height,  # bbox实际的高
            linewidth=1,  # 框的宽度
            edgecolor="r",  # 框的颜色（红）
            facecolor="none",  # 无填充颜色
        )
        # Add the patch to the Axes
        ax.add_patch(rect)  # 在ax对象上叠加一个方形的图案

    plt.show()


def get_bboxes(
        loader,
        model,
        iou_threshold,  # 对于pred_bbox与gt_bbox之间iou的阈值
        threshold,  # 对于概率的阈值
        pred_format="cells",
        box_format="midpoint",
        device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)  # 将数据放入GPU  BS x 448 x 448 x 3
        labels = labels.to(device)  # BS x 7 x 7 x 30（在dataset.py中被设置），但实际上最后一维只有前25个数据有意义

        with torch.no_grad():
            predictions = model(x)  # BS x 7*7*30 predictions和labels都是关于cell归一化的结果

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)  # python list，[BS x S*S x 6]
        bboxes = cellboxes_to_boxes(predictions)  # python list，[BS x S*S x 6]

        for idx in range(batch_size):  # 遍历batch中的每张图片
            nms_boxes = non_max_suppression(
                bboxes[idx],  # 每张图片中的predict_bbox组成的列表 S*S x 6
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )  # nms_boxes为NMS后的bbox的列表

            # if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)  # 在列表最前面加上train_idx维度

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:  # 标签的box[1]是自定义的值非0即1，这里将置1的取出来
                    all_true_boxes.append([train_idx] + box)  # 在相应置信概率超过阈值的标签bbox列表最前加上train_idx维度

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes  # 返回嵌套列表[[train_idx, class_pred, prob_score, x1, y1, x2, y2], ...]


def convert_cellboxes(predictions, S=7):  # BS x S*S*30 ---> BS x S x S x 6
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")  # 将数据传入cpu，下面的cell_indices定义时有.cuda()这里就无需传入cpu
    batch_size = predictions.shape[0]  # predictions形状：BS x 7*7*30
    predictions = predictions.reshape(batch_size, 7, 7, 30)  # reshape成BS x 7 x 7 x 30
    bboxes1 = predictions[..., 21:25]  # BS x 7 x 7 x 4
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(  # 2 x BS x 7 x 7
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0  # 最后一维取出一个数（降维），新增新0维并拼接
    )
    best_box = scores.argmax(0).unsqueeze(-1)  # 返回0维最大值的序号（BS x 7 x 7）并扩充最后一维度 （BS x 7 x 7 x 1）
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2  # 取出两个中概率最大的bbox （BS x 7 x 7 x 4）
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)  # BS x 7 x 7 x 1
    x = 1 / S * (best_boxes[..., :1] + cell_indices)  # 把相对于某个cell归一化的结果转换为相对于整张图像归一化的结果
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))  # 这里的cell_indics是按照行的顺序排列
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)  # BS x 7 x 7 x 4
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)  # 类别概率里最大值的编号 BS x 7 x 7 x 1
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(  # 置信概率里最大的 BS x 7 x 7 x 1
        -1
    )
    converted_preds = torch.cat(  # BS x 7 x 7 x 6, 6分别为类别，概率，4个坐标（相对整幅图像）
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=7):  # out的形状：BS x 7*7*30
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)  # BS x S*S x 6
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):  # 对BS中每个图片遍历
        bboxes = []

        for bbox_idx in range(S * S):  # 对每个cell遍历
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)  # all_bboxes最后shape为：BS x S*S x 6

    return all_bboxes  # 一个batch中所有bboxes（未经NMS），从tensor转为python list


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
