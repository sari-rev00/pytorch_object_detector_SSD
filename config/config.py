class ConfBCCD():
    LIST_LABEL = ["WBC"]


class ConfDataloader():
    IGNORE_NOT_LISTED_LABEL = True
    TARGET_EXT = [".jpg", ".jpeg"]
    TEST_SIZE = 0.2
    BATCH_SIZE = 4


class ConfCnn():
    NUM_CLASSES = len(ConfBCCD.LIST_LABEL) + 1 # add background class
    INPUT_SIZE = 200
    WEIGHT_DATA_DIR = "weight_data"


class ConfDataTransform():
    COLOR_MEAN = (100, 100, 100)


class ConfDefaultBox():
    PERAMS = {
        # 'num_classes': ConfCnn.NUM_CLASSES,
        'input_size': ConfCnn.INPUT_SIZE,
        'bbox_aspect_num': [4, 6, 6, 6, 4],
        'feature_maps': [25, 12, 6, 3, 1],
        'steps': [8, 16, 33, 66, 200], # approximately input_size / feature_map 
        'min_sizes': [20, 40, 60, 80, 120],
        'max_sizes': [40, 60, 80, 120, 160],
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2]]}
    BBOX_ASPECT_NUM = [2 + len(l_ar) * 2 for l_ar in PERAMS['aspect_ratios']]


class ConfTraining():
    LEARNING_RATE = 1e-4
    CLIP_GRAD_VALUE = 2.0
    SGD_MOMENTUM = 0.9
    SGD_WEIGHT_DECAY = 5e-4


class ConfPrediction():
    ANNOTATION_FONT_SIZE = 20
    COLORS = [(150, 0, 0), (0, 150, 0), (0, 0, 150)]


class ConfBoxDetector():
    CONF_TH = 0.8
    TOP_K = 100
    NMS_TH = 0.45
    OVERLAP = 0.45