import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.match import match


class MultiBoxLoss(nn.Module):
    def __init__(self, jaccard_thresh=0.5, neg_pos=3, device='cpu'):
        super(MultiBoxLoss, self).__init__()
        self.jaccard_thresh = jaccard_thresh
        self.negpos_ratio = neg_pos # ratio of data num: neg / pos
        self.device = device

    def forward(self, predictions, targets):
        loc_data, conf_data, dbox_list = predictions

        num_batch = loc_data.size(0)
        num_dbox = dbox_list.size(0)
        num_classes = conf_data.size(2)

        # prepare target(t) data for each dbox --------------------------
        conf_t_label = torch.LongTensor(num_batch, num_dbox).to(self.device)
        loc_t = torch.Tensor(num_batch, num_dbox, 4).to(self.device)
        for idx in range(num_batch):

            truths = targets[idx][:, :-1].to(self.device)
            labels = targets[idx][:, -1].to(self.device)

            dbox = dbox_list.to(self.device)
            variance = [0.1, 0.2]

            match(self.jaccard_thresh, truths, dbox,
                  variance, labels, loc_t, conf_t_label, idx)

        pos_mask = conf_t_label > 0  # torch.Size([num_batch, num_dbox])
        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data) # torch.Size([num_batch, num_dbox, 4])
        loc_t = loc_t[pos_idx].view(-1, 4)

        # loss: location ------
        loc_p = loc_data[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # loss: confidence of classes ------
        batch_conf = conf_data.view(-1, num_classes)
        loss_c = F.cross_entropy(
            batch_conf, conf_t_label.view(-1), reduction='none')
       
        # put loss of positive box into zero ----------
        num_pos = pos_mask.long().sum(1, keepdim=True) 
        loss_c = loss_c.view(num_batch, -1)
        loss_c[pos_mask] = 0

        # Hard Negative Mining -----------
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        num_neg = torch.clamp(num_pos*self.negpos_ratio, max=num_dbox)
        neg_mask = idx_rank < (num_neg).expand_as(idx_rank) # torch.Size([num_batch, num_dbox])

        # pos_mask: torch.Size([num_batch, num_dbox]) 
        # -> pos_idx_mask: torch.Size([num_batch, num_dbox, num_classes])
        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)

        # extruct pos and mined neg data
        conf_hnm = conf_data[
            (pos_idx_mask + neg_idx_mask).gt(0)].view(-1, num_classes)
        conf_t_label_hnm = conf_t_label[(pos_mask + neg_mask).gt(0)]
        loss_c = F.cross_entropy(conf_hnm, conf_t_label_hnm, reduction='sum')

        N = num_pos.sum()
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c
