import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
from matplotlib import pyplot as plt

import parse_cfg
import yolo
import datasets
import utils

cfgfile = './yolo_lite_trial3_no_batch.cfg'
data_config = 'config/coco.data'
epoches = 100
batch_size = 8
n_cpu = 2

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.cuda()
    os.makedirs("output", exist_ok=True)

    model = yolo.Darknet(cfgfile).cuda()
    # print(model)

    optimizer = torch.optim.Adam(model.parameters())

    data_config = parse_cfg.parse_data_config(data_config)

    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = utils.load_classes(data_config["names"])

    dataset = datasets.ListDataset(train_path, augment=True, multiscale=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    # dataiter = iter(dataloader)
    # print(dataiter.next())

    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        print(batch_i)
        print(imgs.size())
        print(targets.size())
        break
    print("img", type(imgs[0]))
    # im = Image.open('image.jpg')
    # imgs[0].show()
    loss, outputs = model(imgs, targets)
    # print(outputs)
    # loss.backward()
    # for epoch in range(epoches):
    #     model.train()
    #     start_time = time.time()

    #     for batch_i, (_, imgs, targets) in enumerate(dataLoader):
    #         batches_done = len(dataLoader)*epoch + batch_i

    #         imgs = Variable(imgs.to(device))
    #         targets = Variable(targets.to(device), requires_grad=False)

    #         loss, outputs = model(imgs, targets)
    #         loss.backward()

    #         if batches_done % opt.gradient_accumulations:
    #             # Accumulates gradient before each step
    #             optimizer.step()
    #             optimizer.zero_grad()
