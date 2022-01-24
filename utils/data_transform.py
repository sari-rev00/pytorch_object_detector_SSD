from config.config import ConfCnn, ConfDataTransform
from ref_utils.data_augumentation import (
    Compose, ConvertFromInts, ToAbsoluteCoords, 
    PhotometricDistort, Expand, RandomSampleCrop, 
    RandomMirror, ToPercentCoords, Resize, SubtractMeans)


INPUT_SIZE = ConfCnn.INPUT_SIZE
COLOR_MEAN = ConfDataTransform.COLOR_MEAN


class ImageDataTransform():
    def __init__(self, input_size=INPUT_SIZE, color_mean=COLOR_MEAN):
        self.data_transform = {
            "train": Compose([
                ConvertFromInts(),
                # ToAbsoluteCoords(),
                # PhotometricDistort(),
                # Expand(mean=color_mean),
                # RandomSampleCrop(),
                # RandomMirror(),
                # ToPercentCoords(),
                Resize(size=input_size),
                SubtractMeans(mean=color_mean)
            ]),
            "val": Compose([
                ConvertFromInts(),
                Resize(size=input_size),
                SubtractMeans(mean=color_mean)
            ])}
        return None
    
    def __call__(self, img, mode, boxes, label_idxs):
        return self.data_transform[mode](img, boxes, label_idxs)
    
