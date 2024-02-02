import torch
import torchvision
import torch.nn.functional as F
import os, time
import numpy as np
from tqdm import tqdm

def load_datafile(data_path):

    cfg = {"model_name":None,
    
           "epochs": None,
           "steps": None,           
           "batch_size": None,
           "subdivisions":None,
           "learning_rate": None,

           "pre_weights": None,        
           "classes": None,
           "width": None,
           "height": None,           
           "anchor_num": None,
           "anchors": None,

           "val": None,           
           "train": None,
           "names":None
        }

    assert os.path.exists(data_path), "请指定正确配置.data文件路径"


    list_type_key = ["anchors", "steps"]
    str_type_key = ["model_name", "val", "train", "names", "pre_weights"]
    int_type_key = ["epochs", "batch_size", "classes", "width",
                   "height", "anchor_num", "subdivisions"]
    float_type_key = ["learning_rate"]
    

    with open(data_path, 'r') as f:
        for line in f.readlines():
            if line == '\n' or line[0] == "[":
                continue
            else:
                data = line.strip().split("=")

                if data[0] in cfg:
                    if data[0] in int_type_key:
                       cfg[data[0]] = int(data[1])
                    elif data[0] in str_type_key:
                        cfg[data[0]] = data[1]
                    elif data[0] in float_type_key:
                        cfg[data[0]] = float(data[1])
                    elif data[0] in list_type_key:
                        cfg[data[0]] = [float(x) for x in data[1].split(",")]
                    else:
                        print("配置文件有错误的配置项")
                else:
                    print("%s配置文件里有无效配置项:%s"%(data_path, data))
    return cfg

def xywh2xyxy(x):

    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:

        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:

        b1_x1, b1_y1, b1_x2, b1_y2 = \
            box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = \
            box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]


    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def compute_ap(recall, precision):

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))


    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])



    i = np.where(mrec[1:] != mrec[:-1])[0]


    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def ap_per_class(tp, conf, pred_cls, target_cls):

    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]


    unique_classes = np.unique(target_cls)


    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()
        n_p = i.sum()

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:

            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()


            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])


            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])


            ap.append(compute_ap(recall_curve, precision_curve))


    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return np.mean(p), np.mean(r), np.mean(ap), np.mean(f1)

def get_batch_statistics(outputs, targets, iou_threshold, device):
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                
                pred_box = pred_box.to(device)
                pred_label = pred_label.to(device)


                if len(detected_boxes) == len(annotations):
                    break


                if pred_label.to(device) not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics

def non_max_suppression(prediction, conf_thres=0.3, iou_thres=0.45, classes=None):

    nc = prediction.shape[2] - 5



    max_wh = 4096
    max_det = 300
    max_nms = 30000
    time_limit = 1.0
    multi_label = nc > 1

    t = time.time()
    output = [torch.zeros((0, 6), device="cpu")] * prediction.shape[0]

    for xi, x in enumerate(prediction):


        x = x[x[..., 4] > conf_thres]


        if not x.shape[0]:
            continue


        x[:, 5:] *= x[:, 4:5]


        box = xywh2xyxy(x[:, :4])


        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]


        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]


        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:

            x = x[x[:, 4].argsort(descending=True)[:max_nms]]


        c = x[:, 5:6] * max_wh

        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]

        output[xi] = x[i].detach().cpu()

        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break

    return output

def make_grid(h, w, cfg, device):
    hv, wv = torch.meshgrid([torch.arange(h), torch.arange(w)])
    return torch.stack((wv, hv), 2).repeat(1,1,3).reshape(h, w, cfg["anchor_num"], -1).to(device)

def handel_preds(preds, cfg, device):

    anchors = np.array(cfg["anchors"])
    anchors = torch.from_numpy(anchors.reshape(len(preds) // 3, cfg["anchor_num"], 2)).to(device)

    output_bboxes = []
    layer_index = [0, 0, 0, 1, 1, 1]

    for i in range(len(preds) // 3):
        bacth_bboxes = []
        reg_preds = preds[i * 3]
        obj_preds = preds[(i * 3) + 1]
        cls_preds = preds[(i * 3) + 2]

        for r, o, c in zip(reg_preds, obj_preds, cls_preds):
            r = r.permute(1, 2, 0)
            r = r.reshape(r.shape[0], r.shape[1], cfg["anchor_num"], -1)

            o = o.permute(1, 2, 0)
            o = o.reshape(o.shape[0], o.shape[1], cfg["anchor_num"], -1)

            c = c.permute(1, 2, 0)
            c = c.reshape(c.shape[0],c.shape[1], 1, c.shape[2])
            c = c.repeat(1, 1, 3, 1)

            anchor_boxes = torch.zeros(r.shape[0], r.shape[1], r.shape[2], r.shape[3] + c.shape[3] + 1)


            grid = make_grid(r.shape[0], r.shape[1], cfg, device)
            stride = cfg["height"] /  r.shape[0]
            anchor_boxes[:, :, :, :2] = ((r[:, :, :, :2].sigmoid() * 2. - 0.5) + grid) * stride


            anchors_cfg = anchors[i]
            anchor_boxes[:, :, :, 2:4] = (r[:, :, :, 2:4].sigmoid() * 2) ** 2 * anchors_cfg


            anchor_boxes[:, :, :, 4] = o[:, :, :, 0].sigmoid()


            anchor_boxes[:, :, :, 5:] = F.softmax(c[:, :, :, :], dim = 3)


            anchor_boxes = anchor_boxes.cpu().detach().numpy() 
            bacth_bboxes.append(anchor_boxes)     


        bacth_bboxes = torch.from_numpy(np.array(bacth_bboxes))
        bacth_bboxes = bacth_bboxes.view(bacth_bboxes.shape[0], -1, bacth_bboxes.shape[-1]) 

        output_bboxes.append(bacth_bboxes)    
        

    output = torch.cat(output_bboxes, 1)
            
    return output

def evaluation(val_dataloader, cfg, model, device, conf_thres = 0.01, nms_thresh = 0.4, iou_thres = 0.5):

    labels = []
    sample_metrics = []
    pbar = tqdm(val_dataloader)

    for imgs, targets in pbar:
        imgs = imgs.to(device).float() / 255.0
        targets = targets.to(device)       


        labels += targets[:, 1].tolist()

        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= torch.tensor([cfg["width"], cfg["height"], cfg["width"], cfg["height"]]).to(device)


        with torch.no_grad():
            preds = model(imgs)


            output = handel_preds(preds, cfg, device)
            output_boxes = non_max_suppression(output, conf_thres = conf_thres, iou_thres = nms_thresh)

        sample_metrics += get_batch_statistics(output_boxes, targets, iou_thres, device)
        pbar.set_description("Evaluation model:") 

    if len(sample_metrics) == 0:
        print("---- No detections over whole validation set ----")
        return None


    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    metrics_output = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    
    return metrics_output     