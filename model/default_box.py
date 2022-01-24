from itertools import product
from math import sqrt
import torch

from config.config import ConfDefaultBox

DBOX_CONFIG = ConfDefaultBox.PERAMS


def default_boxes(cfg=DBOX_CONFIG):
    image_size = cfg['input_size']
    feature_maps = cfg['feature_maps']
    steps = cfg['steps']
    min_sizes = cfg['min_sizes']
    max_sizes = cfg['max_sizes']        
    aspect_ratios = cfg['aspect_ratios']

    mean = list()
    for k, f in enumerate(feature_maps):
        for i, j in product(range(f), repeat=2):
            cx = (j + 0.5) * steps[k] / image_size # cx: 0~1
            cy = (i + 0.5) * steps[k] / image_size # cy: 0~1

            # box: min_sizes
            s_k = min_sizes[k] / image_size
            mean += [cx, cy, s_k, s_k]

            # box: max_sizes
            # s_k_prime = sqrt(s_k * (max_sizes[k] / image_size))
            s_k_prime = max_sizes[k] / image_size
            mean += [cx, cy, s_k_prime, s_k_prime]

            # box: in specified aspect ratio
            for ar in aspect_ratios[k]:
                mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

    output = torch.Tensor(mean).view(-1, 4)
    output.clamp_(max=1, min=0)

    return output
