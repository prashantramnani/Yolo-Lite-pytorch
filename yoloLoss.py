import math
import torch
import torch.nn as nn

class YoloLoss(nn.modules.loss._Loss):
        def __init__(self, num_classes, anchors, coord_scale=1.0, noobject_scale=1,
                object_scale=5, class_scale=1, thresh=0.5):
        super(YoloLoss, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.anchors = torch.Tensor(anchors)

        self.coord_scale = coord_scale
        self.noobject_scale = noobject_scale
        self.object_scale = object_scale
        self.class_scale = class_scale
        self.thresh = thresh

    def forward(self, output, target):
