%matplotlib inline
import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
ROOT_DIR = os.path.abspath("../")
# assert(ROOT_DIR== "/content/Mask_RCNN")
sys.path.append(ROOT_DIR)
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
sys.path.append(os.path.join(ROOT_DIR, "hold")) 
import hold

def inference_init(ROOT_DIR, HOLD_DIR, subset, weights_path, DEVICE = "/gpu:0" ):

    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    MODEL_WEIGHTS_PATH = ROOT_DIR +"/hold_mask_rcnn_coco.h5"

    config = hold.CustomConfig()
    # HOLD_DIR = ROOT_DIR+"/hold/newdata1"

    # Override the training configurations with a few
    # changes for inferencing.
    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        # IMAGE_RESIZE_MODE = 'none'
        # IMAGE_MIN_DIM = 256
        # IMAGE_MAX_DIM = 1024

    config = InferenceConfig()
    config.display()

    # set target device
    # DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

    dataset = hold.CustomDataset()
    # dataset.load_custom(HOLD_DIR, "val")
    dataset.load_custom(HOLD_DIR, subset)
    # Must call before using the dataset
    dataset.prepare()

    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,config=config)

    #weights_path = "../logs/hold20211021T0846/mask_rcnn_hold_0010.h5"
    # weights_path = "/content/drive/MyDrive/ColabNotebooks/logs/hold20211021T0846/mask_rcnn_hold_0010.h5"

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    return model, dataset, config

