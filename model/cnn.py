import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from config.config import ConfCnn, ConfDefaultBox
from model.default_box import default_boxes


BBOX_ASPECT_NUM = ConfDefaultBox.BBOX_ASPECT_NUM
NUM_CLASSES = ConfCnn.NUM_CLASSES
CHANNEL_NUM = [256, 512, 256, 128, 128]
WEIGHT_DATA_DIR = ConfCnn.WEIGHT_DATA_DIR


def initialize_weight(instance):
    for m in instance.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', 
                nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
    return None


class SSD(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg200 = Vgg200()
        self.l2norm = L2Norm()
        self.extras = Extras()
        self.loc = BoxLocation()
        self.conf = BoxConfidence()
        self.dbox_list = default_boxes()
        return None
    
    def forward(self, x):
        pre_source_1, source_2 = self.vgg200(x)
        source_1 = self.l2norm(pre_source_1)
        source_3, source_4, source_5 = self.extras(source_2)
        loc_out = self.loc(
            source_1=source_1, 
            source_2=source_2, 
            source_3=source_3, 
            source_4=source_4, 
            source_5=source_5)   
        conf_out = self.conf(
            source_1=source_1, 
            source_2=source_2, 
            source_3=source_3, 
            source_4=source_4, 
            source_5=source_5)
        return (loc_out, conf_out, self.dbox_list)
    
    def save_weight(self, fname):
        if not ".pth" in fname:
            fname += ".pth"
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(WEIGHT_DATA_DIR, fname))
        return None
    
    def load_weight(self, fname):
        if not ".pth" in fname:
            fname += ".pth"
        state_dict = torch.load(os.path.join(WEIGHT_DATA_DIR, fname))
        self.load_state_dict(state_dict)
        return None


class Vgg200(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_front, self.net_back = self.build_network()
        initialize_weight(instance=self)
        return None

    def build_network(self):
        front = nn.Sequential(
            # [200 x 200 x 3]
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [100 x 100 x 32]
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [50 x 50 x 64]
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [25 x 25 x 128]
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True)
            # [25 x 25 x 256]
        )
        back = nn.Sequential(
            # [25 x 25 x 256]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [12 x 12 x 256]
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True)
            # [12 x 12 x 512]
        )
        return front, back

    def forward(self, x):
        pre_source_1 = self.net_front(x)
        source_2 = self.net_back(pre_source_1)
        return pre_source_1, source_2


class Extras(nn.Module):
    def __init__(self):
        super().__init__()
        self.unit_1, self.unit_2, self.unit_3 = self.build_network()
        initialize_weight(instance=self)
        return None
    
    def build_network(self):
        unit_1 = nn.Sequential(
            # [12 x 12 x 512]
            nn.Conv2d(512, 128, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2), nn.ReLU(inplace=True),
            # [6 x 6 x 256]
        )
        unit_2 = nn.Sequential(
            # [6 x 6 x 256]
            nn.Conv2d(256, 64, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2), nn.ReLU(inplace=True),
            # [3 x 3 x 128]
        )
        unit_3 = nn.Sequential(
            # [3 x 3 x 128]
            nn.Conv2d(128, 64, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3), nn.ReLU(inplace=True)
            # [1 x 1 x 128]
        )
        return unit_1, unit_2, unit_3
    
    def forward(self, source_2):
        source_3 = self.unit_1(source_2)
        source_4 = self.unit_2(source_3)
        source_5 = self.unit_3(source_4)
        return source_3, source_4, source_5
    

class BoxLocation(nn.Module):
    def __init__(
            self, 
            list_ch=CHANNEL_NUM, 
            bbox_aspect_num=BBOX_ASPECT_NUM):
        super().__init__()
        self.list_ch = list_ch
        self.bbox_aspect_num = bbox_aspect_num
        self.net_box_location = self.build_network(list_ch, bbox_aspect_num)
        initialize_weight(instance=self)
        return None
    
    def build_network(self, list_ch, bbox_aspect_num):
        layers = list()
        for ch, asp in zip(list_ch, bbox_aspect_num):
            layers.append(nn.Conv2d(ch, asp * 4, kernel_size=3, padding=1))
        return nn.ModuleList(layers)
    
    def forward(self, source_1, source_2, source_3, source_4, source_5):
        loc = list()
        list_source = [source_1, source_2, source_3, source_4, source_5]
        for s, l in zip(list_source, self.net_box_location):
            loc.append(l(s).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        return loc.view(loc.size(0), -1, 4)


class BoxConfidence(nn.Module):
    def __init__(
            self, 
            list_ch=CHANNEL_NUM, 
            bbox_aspect_num=BBOX_ASPECT_NUM, 
            num_classes=NUM_CLASSES):
        super().__init__()
        self.list_ch = list_ch
        self.bbox_aspect_num = bbox_aspect_num
        self.num_classes = num_classes
        self.net_box_confidence = self.build_network(list_ch, bbox_aspect_num, num_classes)
        initialize_weight(instance=self)
        return None
    
    def build_network(self, list_ch, bbox_aspect_num, num_classes):
        layers = list()
        for ch, asp in zip(list_ch, bbox_aspect_num):
            layers.append(nn.Conv2d(ch, asp * num_classes, kernel_size=3, padding=1))
        return nn.ModuleList(layers)
    
    def forward(self, source_1, source_2, source_3, source_4, source_5):
        conf = list()
        list_source = [source_1, source_2, source_3, source_4, source_5]
        for s, c in zip(list_source, self.net_box_confidence):
            conf.append(c(s).permute(0, 2, 3, 1).contiguous())
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        return conf.view(conf.size(0), -1, self.num_classes)
        
        
class L2Norm(nn.Module):
    def __init__(self, input_channels=256, scale=20):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale
        self.reset_parameters()
        self.eps = 1e-10
        return None

    def reset_parameters(self):
        init.constant_(self.weight, self.scale)
        return None

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x, norm)
        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        return weights * x