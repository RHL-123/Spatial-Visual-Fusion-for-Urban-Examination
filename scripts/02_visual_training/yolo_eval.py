import os
import torch
import numpy as np
from tqdm import tqdm
from yolo_metrics import box_iou, ap_per_class
from torchvision.ops import box_convert

def evaluate_yolo(model, data_loader, device, class_names, iou_thr=0.5, conf_thr=0.05):
    model.eval()
    stats = []  # (tp, conf, pred_cls, target_cls)
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc='Eval', ncols=80):
            images = torch.stack(images).to(device)
            outputs = model(images)
            pred_logits = outputs['pred_logits']
            pred_boxes = outputs['pred_boxes']
            # 取最后一层输出
            pred_logits = pred_logits[:, -1, :, :]
            pred_boxes = pred_boxes[:, -1, :, :]
            for i in range(images.shape[0]):
                pred_prob = pred_logits[i].softmax(-1)
                pred_conf, pred_cls = pred_prob.max(-1)
                pred_box = pred_boxes[i].cpu().numpy()
                pred_cls = pred_cls.cpu().numpy()
                pred_conf = pred_conf.cpu().numpy()
                # 过滤背景
                keep = (pred_cls > 0) & (pred_conf > conf_thr)
                pred_box = pred_box[keep]
                pred_cls = pred_cls[keep]
                pred_conf = pred_conf[keep]
                # 转为xyxy
                pred_box = box_convert(torch.tensor(pred_box), in_fmt='cxcywh', out_fmt='xyxy').numpy() if pred_box.shape[1]==4 else pred_box
                gt_box = targets[i]['boxes'].cpu().numpy()
                gt_cls = targets[i]['labels'].cpu().numpy()
                detected = []
                tps = np.zeros(len(pred_box))
                for j, pb in enumerate(pred_box):
                    ious = box_iou(np.expand_dims(pb, 0), gt_box)
                    max_iou_idx = ious.argmax(1)
                    max_iou = ious[0, max_iou_idx]
                    if max_iou > iou_thr and pred_cls[j] in gt_cls and max_iou_idx not in detected:
                        tps[j] = 1
                        detected.append(max_iou_idx[0])
                stats.append((tps, pred_conf, pred_cls, gt_cls))
    # 汇总
    stats = [np.concatenate(x, 0) for x in zip(*stats)] if stats else ([], [], [], [])
    ap, precision, recall, f1, unique_classes = ap_per_class(*stats, num_classes=len(class_names))
    results = {
        'mAP@0.5': ap.mean() if len(ap) else 0,
        'precision': precision.mean() if len(precision) else 0,
        'recall': recall.mean() if len(recall) else 0,
        'f1': f1.mean() if len(f1) else 0,
        'per_class_ap': dict(zip([class_names[c-1] for c in unique_classes if c>0], ap)),
        'per_class_precision': dict(zip([class_names[c-1] for c in unique_classes if c>0], precision)),
        'per_class_recall': dict(zip([class_names[c-1] for c in unique_classes if c>0], recall)),
    }
    return results

