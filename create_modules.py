import yolo_layer   

import torch
import torch.nn as nn

def create_modules(blocks):
    module_list = nn.ModuleList()
    filters = 3
    prev_filters = 3
    index = 0
    for block in blocks:
        module = nn.Sequential()
        
        
        output_filters = []


        if  block["type"] == "net":
            hyperparams = block
            # print(hyperparams)

        elif  block["type"] == "convolutional":
            activation = block["activation"]

            try:
                batch_normalize = int(block["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True 

            filters = int(block["filters"])
            kernel_size = int(block["size"])
            stride = int(block["stride"])
            pad = int(block["pad"])

            if pad:
                pad = kernel_size//2
            else:
                pad = 0
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)

        elif block["type"] == "maxpool":
            stride = int(block["stride"])
            size = int(block["size"])
            if stride != 1:
                maxpool = nn.MaxPool2d(size, stride)
            else:
                maxpool = MaxPoolStride1(size)
            
            module.add_module("maxpool_{0}".format(index), maxpool)

        elif block["type"] == "region":
            # print(block)
            anchors = block["anchors"].split(",")
            anchors = [float(a) for a in anchors]  
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)] 
            num_classes = int(block["classes"]) 
            img_size = int(hyperparams["height"])

            yoloLayer = yolo_layer.YoloLayer(anchors, num_classes, img_size)        

            module.add_module("region_{0}".format(index), yoloLayer)    
            
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        index += 1 

            
    return hyperparams, module_list     