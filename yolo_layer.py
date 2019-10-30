import torch
import torch.nn as nn

class YoloLayer(nn.Module):
    """ Detection Layer """

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YoloLayer, self).__init__()

        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.threshold = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self ,grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size

        self.stride = self.img_dim / self.grid_size

        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)

        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) 
            for a_w, a_h in self.anchors])

        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):

        return 1, 2

        FloatTensor = torch.cuda.FloatTensor if x.is_cuds else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuds else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuds else torch.ByteTensor      

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)     

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0,1,3,4,2).contiguous()
        )


        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid([prediction[..., 1]])
        w = torch.sigmoid([prediction[..., 2]])
        h = torch.sigmoid([prediction[..., 3]])
        box_confidence = torch.sigmoid([prediction[..., 4]])
        class_score = torch.sigmoid(prediction[..., 5:])


        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data)*self.anchor_w 
        pred_boxes[..., 3] = torch.exp(h.data)*self.anchor_h


        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets == None:
            return outputs
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                threshold=self.threshold,
            )    
