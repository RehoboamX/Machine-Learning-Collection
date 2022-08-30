import torch
from collections import Counter
from iou import intersection_over_union

def mean_average_precision(
        pred_boxes, true_boxes, iou_threshold=0.5, box_format="corners", num_classes=20
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

    # pred_boxes (list): [[train_idx, class_pred, prob_score, x1, y1, x2, y2], ...]
    average_precisions = []
    epsilon = 1e-6  # epsilon的定义目的是保持数值稳定性

    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)    # 将所有c类别的预测bboxes放入detections列表

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)   # 将所有c类别的标签bboxes放入ground_truths列表

        # img 0 has 3 bboxes
        # img 1 has 5 bboxes
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])    # counter括号中的部分就是单独把train_idx拿出来的数组

        for key, val in amount_bboxes.items():  # 遍历python字典键值对时要加.items()
            amount_bboxes[key] = torch.zeros(val)  # amount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}

        detections.sort(key=lambda x: x[2], reverse=True)   # 将detections中的元素按照prob_score降序排列
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]   # 找出与该detection框照片序号
                # train_idx相同的ground_truths中的元素放入ground_truth_img列表中
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(   # 计算该detection框和同一图片中所有ground_truth_img框(同类别)的IOU
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx   # 找到该detection与之最大IOU的ground_truth在ground_truth_img列表中的序号

            if best_iou > iou_threshold:  # 证明这个prediction框是正确的
                if amount_bboxes[detection[0]][best_gt_idx] == 0:  # 检查之前是否有cover过这个ground_truth的bbox
                    TP[detection_idx] = 1  # 将TP列表中该样本的位置置1
                    amount_bboxes[detection[0]][best_gt_idx] = 1  # 标记该ground_truth的bbox已经被cover
                else:  # 若找到的ground_truth的bbox已经被cover过
                    FP[detection_idx] = 1   # 将FP列表中该样本的位置置1

            else:   # IOU没有高于设置的临界值
                FP[detection_idx] = 1

        # [1, 1, 0, 1, 0] -> [1, 2, 2, 3, 3]
        TP_cumsum = torch.cumsum(TP, dim=0)  # 计算recall值时有用
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)  # total_true_bboxes = len(ground_truths)
        precisions = torch.devide(TP_cumsum + FP_cumsum + epsilon)  # torch.devide()是元素级别的除法
        precisions = torch.cat((torch.tensor([1]), precisions))  # 在precision首端添加元素1，因为recall-precisions坐标系的第一个点要为（0,1）
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))  # torch.trapz(y, x)计算x-y坐标系下坐标点与x轴围成的面积大小

    return sum(average_precisions) / len(average_precisions)
