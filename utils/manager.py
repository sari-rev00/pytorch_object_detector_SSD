from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim

from config.config import ConfTraining, ConfBoxDetector, ConfCnn, ConfBCCD, ConfPrediction
from utils.box_detector import Detect
from utils.data_transform import ImageDataTransform
from utils.loss_function import MultiBoxLoss


LEARNING_RATE = ConfTraining.LEARNING_RATE
CLIP_GRAD_VALUE = ConfTraining.CLIP_GRAD_VALUE
MOMENTUM = ConfTraining.SGD_MOMENTUM
WEIGHT_DECAY = ConfTraining.SGD_WEIGHT_DECAY

NUM_CLASSES = ConfCnn.NUM_CLASSES
CONF_TH = ConfBoxDetector.CONF_TH
TOP_K = ConfBoxDetector.TOP_K
NMS_TH = ConfBoxDetector.NMS_TH
OVERLAP = ConfBoxDetector.OVERLAP

ANNOTATION_FONT_SIZE = ConfPrediction.ANNOTATION_FONT_SIZE
LIST_COLOR = ConfPrediction.COLORS


class Manager():
    def __init__(self, model):
        self.model = model
        self.load_method()
        return None
    
    def train(
            self,
            num_epochs, 
            dataloader, 
            optimizer=None, 
            criterion=None, 
            auto_save=True,
            print_iter=True):
        if not optimizer:
            optimizer = optim.SGD(
                self.model.parameters(), 
                lr=LEARNING_RATE,
                momentum=MOMENTUM, 
                weight_decay=WEIGHT_DECAY)
        if not criterion:
            criterion = MultiBoxLoss()
        iteration = 1
        logs = list()
        validation = False
        for epoch in range(1, num_epochs + 1):
            epoch_train_loss_l = 0.0
            epoch_train_loss_c = 0.0
            epoch_val_loss_l = 0.0
            epoch_val_loss_c = 0.0
            t_epoch_start = time.time()
            t_iter_start = time.time()
            if((epoch % 10) == 0):
                validation = True
            print('Epoch {}/{} ----------------'.format(epoch, num_epochs))
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    if validation:
                        self.model.eval()
                    else:
                        continue
                dataloader.set_mode(mode=phase)
                for images, targets in dataloader: # batches
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(images)
                        # print(type(outputs))

                        # loc_data, conf_data, dbox_list = outputs
                        # print(loc_data.shape)
                        # print(conf_data.shape)

                        loss_l, loss_c = criterion(outputs, targets)
                        loss = loss_l + loss_c
                        if phase == 'train':
                            loss.backward()
                            nn.utils.clip_grad_value_(
                                self.model.parameters(), 
                                clip_value=CLIP_GRAD_VALUE)
                            optimizer.step()
                            if (iteration % 10 == 0):
                                t_iter_finish = time.time()
                                duration = t_iter_finish - t_iter_start
                                if print_iter:
                                    print('    iteration {} | Loss_c: {:.4f} | Loss_l: {:.4f}'.format(
                                        iteration, loss_c.item(), loss_l.item()))
                                t_iter_start = time.time()
                            epoch_train_loss_l += loss_l.item()
                            epoch_train_loss_c += loss_c.item()
                            iteration += 1
                        else:
                            epoch_val_loss_l += loss_l.item()
                            epoch_val_loss_c += loss_c.item()
            t_epoch_finish = time.time()
            print('    TRAIN | epoch_loss_l: {:.4f}, epoch_loss_c: {:.4f}'.format(
                epoch_train_loss_l, epoch_train_loss_c))
            if validation:
                print('    VAL | epoch_loss_l: {:.4f}, epoch_loss_c: {:.4f}'.format(
                    epoch_val_loss_l, epoch_val_loss_c))
            print('    batch elasped time:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
            t_epoch_start = time.time()
            log_epoch = {
                'epoch': epoch,
                'train_loss_l': epoch_train_loss_l,
                'train_loss_c': epoch_train_loss_c,
                'val_loss_l': None,
                'val_loss_c': None}
            if validation:
                log_epoch.update({
                    'val_loss_l': epoch_val_loss_l,
                    'val_loss_c': epoch_val_loss_l})
            logs.append(log_epoch)
            if validation:
                validation = False
        return logs
    
    def load_method(self, img_transform=None, box_detector=None):
        if img_transform:
            self.img_transform = img_transform
        else:
            self.img_transform = ImageDataTransform()
        if box_detector:
            self.box_detector = box_detector
        else:
            self.box_detector = Detect()
        return None

    def predict(self, img_pil):
        width, height = img_pil.size
        img, _, _ = self.img_transform(
            img=np.array(img_pil), 
            mode="val", 
            boxes=None, 
            label_idxs=None)
        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)
        img = torch.unsqueeze(img, 0)

        self.model.eval()
        loc, conf, dbox = self.model(img)
        boxes_labels = self.box_detector(loc, conf, dbox).detach().numpy()
        boxes_labels = boxes_labels.reshape(NUM_CLASSES, TOP_K, 5)
        boxes_labels = boxes_labels[1:, :, :]

        list_box = list()
        list_label = list()
        list_conf = list()
        coef = np.array([width, height, width, height])

        for i in range(NUM_CLASSES - 1):
            for j in range(TOP_K):
                conf = boxes_labels[i, j, 0]
                if conf >= 0.23:
                    list_label.append(ConfBCCD.LIST_LABEL[i])
                    list_box.append((boxes_labels[i, j, 1:] * coef).astype(np.int16))
                    list_conf.append(boxes_labels[i, j, 0])
        return list_box, list_label, list_conf

    def predict_from_img_file(self, fpath):
        return self.predict(Image.open(fpath))
    
    def annotated_img(self, img_pil, list_box, list_label, list_conf):
        dict_label_color=dict([(k, v) for k, v in zip(list_label, LIST_COLOR)])

        def set_text(text, origin, draw, color):
            text = " " + text + " "
            font = ImageFont.truetype("arial.ttf", size=ANNOTATION_FONT_SIZE)
            txw, txh = draw.textsize(text, font=font)
            draw.rectangle(
                (origin[0], origin[1], origin[0]+txw, origin[1]+txh), 
                outline=color,
                fill=color)
            draw.text(origin, text, fill=(255, 255, 255), font=font)
            return None

        draw = ImageDraw.Draw(img_pil)
        for box, label, conf in zip(list_box, list_label, list_conf):
            color = dict_label_color[label]
            conf = round(conf * 100, 1)
            xmin, ymin, xmax, ymax = box[:4]
            draw.rectangle(
                (xmin, ymin, xmax, ymax), 
                outline=color)
            set_text(
                text="{} {}".format(label, conf), 
                origin=(xmin, ymin), 
                draw=draw, 
                color=color)
        return img_pil
    
    def annotated_img_from_file(self, fpath):
        img_pil=Image.open(fpath)
        list_box, list_label, list_conf = self.predict(img_pil)
        annotated_img = self.annotated_img(
            img_pil=img_pil, 
            list_box=list_box, 
            list_label=list_label, 
            list_conf=list_conf)
        return annotated_img



        
    
