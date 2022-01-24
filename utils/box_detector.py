from torch.autograd import Function
import torch
import torch.nn as nn

from config.config import ConfBoxDetector


CONF_TH = ConfBoxDetector.CONF_TH
TOP_K = ConfBoxDetector.TOP_K
NMS_TH = ConfBoxDetector.NMS_TH
OVERLAP = ConfBoxDetector.OVERLAP


def decode(loc, dbox_list):
    boxes = torch.cat((
        dbox_list[:, :2] + loc[:, :2] * 0.1 * dbox_list[:, 2:],
        dbox_list[:, 2:] * torch.exp(loc[:, 2:] * 0.2)), 
        dim=1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return torch.clamp(boxes, min=0, max=1) # xmin, ymin, xmax, ymax: 0~1


def nm_suppression(boxes, scores, overlap=OVERLAP, top_k=TOP_K):
    count = 0
    keep = scores.new(scores.size(0)).zero_().long()

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)

    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    v, idx = scores.sort(0)
    idx = idx[-top_k:]

    while idx.numel() > 0:
        i = idx[-1]

        keep[count] = i
        count += 1

        if idx.size(0) == 1:
            break

        idx = idx[:-1]
        torch.index_select(x1, 0, idx, out=tmp_x1)
        torch.index_select(y1, 0, idx, out=tmp_y1)
        torch.index_select(x2, 0, idx, out=tmp_x2)
        torch.index_select(y2, 0, idx, out=tmp_y2)
        tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, max=y2[i])

        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)

        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1

        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)

        inter = tmp_w*tmp_h

        rem_areas = torch.index_select(area, 0, idx)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union
        idx = idx[IoU.le(overlap)]

    return keep, count


class Detect(Function):
    def __init__(self, conf_thresh=CONF_TH, top_k=TOP_K, nms_thresh=NMS_TH):
        self.softmax = nn.Softmax(dim=-1)
        self.conf_thresh = conf_thresh
        self.top_k = top_k
        self.nms_thresh = nms_thresh

    def forward(self, loc_data, conf_data, dbox_list):
        num_batch = loc_data.size(0)
        num_dbox = loc_data.size(1)
        num_classes = conf_data.size(2)

        conf_data = self.softmax(conf_data)
        output = torch.zeros(num_batch, num_classes, self.top_k, 5)
        conf_preds = conf_data.transpose(2, 1)

        for i in range(num_batch):
            decoded_boxes = decode(loc_data[i], dbox_list)
            conf_scores = conf_preds[i].clone()
            for cl in range(1, num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.nelement() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                ids, count = nm_suppression(
                    boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = torch.cat(
                    (scores[ids[:count]].unsqueeze(1),
                    boxes[ids[:count]]), 
                    1)
        return output


