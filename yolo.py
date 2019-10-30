import parse_cfg
import create_modules

import torch
import torch.nn as nn

     
class Darknet(nn.Module):
    def __init__(self, cfgfile, img_size=416):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg.parse_cfg(cfgfile)
        self.hyperparams, self.module_list = create_modules.create_modules(self.blocks)
        self.img_size = img_size

    def forward(self, x, targets=None):
        img_dim = x.shape[2]
        loss = 0

        layer_outputs, yolo_outputs = [], []
        
        for i, (blocks, module) in enumerate(zip(self.blocks, self.module_list)):
            print(i)
            if blocks["type"] in ["convolutional", "maxpool"]:
                x = module(x)
                print(x.size())
        #     elif blocks["type"] == "region":
        #         x, layer_loss = module[0](x, targets, img_dim)
        #         loss += layer_loss
        #         yolo_outputs.append(x)
        #     layer_outputs.append(x)
        # yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        # return 1, 2
        return 1, x
        return yolo_outputs if targets is None else (loss, yolo_outputs)




# cfgfile = './yolo_lite_trial3_no_batch.cfg'
# net = Darknet(cfgfile)
# modules = net.blocks[1:]
# for i in range(len(net.blocks)):
#     module_type = net.blocks[i]["type"]
#     if module_type == "convolutional" or module_type == "maxpool":
#         print(net.module_list[i])
