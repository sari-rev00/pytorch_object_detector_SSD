from pprint import pprint
import numpy as np
import os
from PIL import Image, ImageFilter, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data as data
import torchvision
import xml.etree.ElementTree as ET

from config.config import ConfBCCD, ConfDataloader
from utils.data_transform import ImageDataTransform


LIST_LABEL = ConfBCCD.LIST_LABEL
IGNORE_NOT_LISTED_LABEL = ConfDataloader.IGNORE_NOT_LISTED_LABEL
TARGET_EXT = ConfDataloader.TARGET_EXT
TEST_SIZE = ConfDataloader.TEST_SIZE
BATCH_SIZE = ConfDataloader.BATCH_SIZE


def bccd_data_path_list(bccd_dir=""):
    img_dir = os.path.join(bccd_dir, "BCCD_Dataset", "BCCD", "JPEGImages")
    list_fname = list()
    for fname in os.listdir(img_dir):
        for ext in TARGET_EXT:
            if ext in fname:
                list_fname.append(fname)
    xml_dir = os.path.join(bccd_dir, "BCCD_Dataset", "BCCD", "Annotations")
    list_img_path = [os.path.join(img_dir, fname) for fname in list_fname]
    list_xml_path = [os.path.join(xml_dir, fname.split(".")[0] + ".xml") for fname in list_fname]
    # list_xml_data = [normalized_annottion_data(fpath=fpath) for fpath in list_xml_path]
    # return list_img_path, list_xml_data
    return list_img_path, list_xml_path


def normalized_annottion_data(fpath, list_label=None, digits=2):
    tree = ET.parse(fpath)
    list_anno_data = list()
    for elem in tree.iter():
        if 'size'in elem.tag:
            for attr in list(elem):
                if 'width'in attr.tag:
                    w = int(attr.text)
                elif 'height'in attr.tag:
                    h = int(attr.text)
            break
    for elem in tree.iter():
        if 'object' in elem.tag:
            for attr in list(elem):
                if 'name' in attr.tag:
                    name = attr.text
                elif 'bndbox' in attr.tag:
                    for dim in list(attr):
                        if 'xmin' in dim.tag:
                            xmin = round(int(dim.text) / w, digits)
                        elif 'ymin' in dim.tag:
                            ymin = round(int(dim.text) / h, digits)
                        elif 'xmax' in dim.tag:
                            xmax = round(int(dim.text) / w, digits)
                        elif 'ymax' in dim.tag:
                            ymax = round(int(dim.text) / h, digits)
            if list_label:
                if not name in list_label:
                    if IGNORE_NOT_LISTED_LABEL:
                        continue
                    else:
                        raise ValueError(
                            "Error: label {} is not in specified label list {}".format(
                                name,
                                list_label))
                list_anno_data.append(
                    [xmin, ymin, xmax, ymax, list_label.index(name)])
                    # label index 0 means background, object indeces must start from 1.
            else:
                list_anno_data.append([xmin, ymin, xmax, ymax, name])
    return np.array(list_anno_data)
            

class Dataset(data.Dataset):
    def __init__(
            self, 
            list_img_path, 
            list_xml_path, 
            transform_img,
            mode="train", 
            test_size=TEST_SIZE,
            xml_decoder=normalized_annottion_data,
            list_label=None):
        self.list_img_path = list_img_path
        self.list_label = list_label
        self.transform_img = transform_img
        self.list_anno_data = self.prepare_anno_data_list(
            list_xml_path=list_xml_path,
            xml_decoder=xml_decoder)
        self.drop_invalid_data()
        self.split_index = self.split_index(test_size=test_size)
        self.set_mode(mode=mode)
        return None
    
    def __len__(self):
        if self.mode == "train":
            return len(self.list_img_path[:self.split_index])
        else:
            return len(self.list_img_path[self.split_index:])
    
    def __getitem__(self, index):
        if self.mode == "val":
            index += self.split_index
        img = np.array(Image.open(self.list_img_path[index]))
        # height, width, channels = img.shape
        annotations = self.list_anno_data[index]
        img, boxes, labels = self.transform_img(
            img=img, 
            mode=self.mode, 
            boxes=annotations[:, :4],
            label_idxs=annotations[:, 4])
        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)
        boxes_labels = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return img, boxes_labels
    
    def set_mode(self, mode):
        if mode == "val":
            self.mode = "val"
        else:
            self.mode = "train"
        return self
    
    def get_mode(self):
        if self.mode == "train":
            return "train"
        else:
            return "val"
    
    def split_index(self, test_size):
        if not (0 < test_size < 1):
            raise ValueError(
                f"Error: test_size must be 0 < test_size < 1 but {test_size} is assigned.")
        return int(len(self.list_img_path) * (1 - test_size))
    
    def prepare_anno_data_list(self, list_xml_path, xml_decoder):
        return [xml_decoder(
            fpath=fpath,
            list_label=self.list_label) for fpath in list_xml_path]
    
    def drop_invalid_data(self):
        list_i = list()
        list_a = list()
        for i, a in zip(self.list_img_path, self.list_anno_data):
            if 0 < len(a):
                list_i.append(i)
                list_a.append(a)
        self.list_img_path = list_i
        self.list_anno_data = list_a
        return None

    def img_file_name(self, index):
        if self.mode == "val":
            index += self.split_index
        return self.list_img_path[index]
    
    def getitem_as_annotated_img(self, index, dict_label_color):
        if self.mode == "val":
            index += self.split_index
        img = Image.open(self.list_img_path[index])
        w, h = img.size
        draw = ImageDraw.Draw(img)
        for box in self.list_anno_data[index]:
            xmin, ymin, xmax, ymax = box[:4]
            xmin *= w
            xmax *= w
            ymin *= h
            ymax *= h
            if self.list_label:
                name = self.list_label[int(box[4])]
            else:
                name = box[4]
            draw.rectangle(
                (xmin, ymin, xmax, ymax), 
                outline=dict_label_color[name])
            self.set_text(
                text=name, 
                origin=(xmin, ymin), 
                draw=draw, 
                color=dict_label_color[name])
        img.show()
        return None
        
    def set_text(self, text, origin, draw, color):
        text = " " + text + " "
        font = ImageFont.truetype("arial.ttf", size=10)
        txw, txh = draw.textsize(text, font=font)
        draw.rectangle(
            (origin[0], origin[1], origin[0]+txw, origin[1]+txh), 
            outline=(255, 0, 0),
            fill=color)
        draw.text(origin, text, fill=(255, 255, 255))
        return None


def collate_fn(batch):
    targets = list()
    imgs = list()
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    imgs = torch.stack(imgs, dim=0)
    return imgs, targets


class Dataloader(data.DataLoader):
    def __init__(self, dataset, batch_size, mode="train", shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.set_mode(mode=mode)
        super().__init__(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=collate_fn)
        return None
    
    def set_mode(self, mode):
        if mode == "val":
            self.dataset.set_mode(mode="val")
        else:
            self.dataset.set_mode(mode="train")
        return self
    
    def get_mode(self):
        return self.dataset.get_mode()
    

def gen_bccd_dataloader(
        list_label=LIST_LABEL, 
        batch_size=BATCH_SIZE,
        shuffle=True):
    list_img_path, list_xml_path = bccd_data_path_list()
    transform_img = ImageDataTransform()
    ds = Dataset(
        list_img_path=list_img_path, 
        list_xml_path=list_xml_path, 
        transform_img=transform_img,
        list_label=list_label)
    return Dataloader(
        dataset=ds, 
        batch_size=batch_size, 
        shuffle=shuffle)
