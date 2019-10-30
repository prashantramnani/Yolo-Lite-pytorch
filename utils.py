import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import  Variable 
import numpy as np

def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names

def build_targets(pred_boxes, pred_cls, target, anchors, threshold):

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    numberOfBatches = pred_boxes.size(0)
    numberOfAnchors = pred_boxes.size(1)
    numberOfClasses = pred_cls.size(-1)
    numberGridSize = pred_boxes.size(2)



