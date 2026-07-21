import numpy as np

def box_iou(box1, box2):
    """
    计算IoU，输入为[x1, y1, x2, y2]格式
    box1: [N, 4]
    box2: [M, 4]
    返回: [N, M] IoU矩阵
    """
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    iou = np.zeros((box1.shape[0], box2.shape[0]))
    for i in range(box1.shape[0]):
        for j in range(box2.shape[0]):
            xx1 = max(box1[i, 0], box2[j, 0])
            yy1 = max(box1[i, 1], box2[j, 1])
            xx2 = min(box1[i, 2], box2[j, 2])
            yy2 = min(box1[i, 3], box2[j, 3])
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            inter = w * h
            union = area1[i] + area2[j] - inter
            iou[i, j] = inter / union if union > 0 else 0
    return iou

def ap_per_class(tp, conf, pred_cls, target_cls, iou_thr=0.5, num_classes=7):
    """
    计算每个类别的AP、Precision、Recall
    tp: [N] 预测为正的标记（0/1）
    conf: [N] 置信度
    pred_cls: [N] 预测类别
    target_cls: [N] 真实类别
    返回: ap, precision, recall, f1, unique_classes
    """
    # 按置信度排序
    idx = np.argsort(-conf)
    tp, conf, pred_cls = tp[idx], conf[idx], pred_cls[idx]
    unique_classes = np.unique(np.concatenate((pred_cls, target_cls)))
    ap, precision, recall, f1 = [], [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()
        n_p = i.sum()
        if n_p == 0 or n_gt == 0:
            ap.append(0)
            precision.append(0)
            recall.append(0)
            f1.append(0)
            continue
        # 累积TP/FP
        fpc = (1 - tp[i]).cumsum()
        tpc = tp[i].cumsum()
        recall_curve = tpc / (n_gt + 1e-16)
        precision_curve = tpc / (tpc + fpc)
        # AP计算（11点插值）
        ap_c = 0
        for t in np.linspace(0, 1, 11):
            p = precision_curve[recall_curve >= t]
            ap_c += p.max() if p.size else 0
        ap_c /= 11
        ap.append(ap_c)
        precision.append(precision_curve[-1])
        recall.append(recall_curve[-1])
        f1.append(2 * precision_curve[-1] * recall_curve[-1] / (precision_curve[-1] + recall_curve[-1] + 1e-16))
    return np.array(ap), np.array(precision), np.array(recall), np.array(f1), unique_classes

