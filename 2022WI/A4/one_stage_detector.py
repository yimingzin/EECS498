import math
from typing import Dict, List, Optional

import torch
from a4_helper import *
from common import DetectorBackboneWithFPN, class_spec_nms, get_fpn_location_coords
from torch import nn
from torch.nn import functional as F
from torch.utils.data._utils.collate import default_collate
from torchvision.ops import sigmoid_focal_loss

# Short hand type notation:
TensorDict = Dict[str, torch.Tensor]


def hello_one_stage_detector():
    print("Hello from one_stage_detector.py!")


class FCOSPredictionNetwork(nn.Module):
    """
    FCOS prediction network that accepts FPN feature maps from different levels
    and makes three predictions at every location: bounding boxes, class ID and
    centerness. This module contains a "stem" of convolution layers, along with
    one final layer per prediction. For a visual depiction, see Figure 2 (right
    side) in FCOS paper: https://arxiv.org/abs/1904.01355

    We will use feature maps from FPN levels (P3, P4, P5) and exclude (P6, P7).
    """

    def __init__(
        self, num_classes: int, in_channels: int, stem_channels: List[int]
    ):
        """
        Args:
            num_classes: Number of object classes for classification.
            in_channels: Number of channels in input feature maps. This value
                is same as the output channels of FPN, since the head directly
                operates on them.
            stem_channels: List of integers giving the number of output channels
                in each convolution layer of stem layers.
        """
        super().__init__()

        ######################################################################
        # TODO: Create a stem of alternating 3x3 convolution layers and RELU
        # activation modules. Note there are two separate stems for class and
        # box stem. The prediction layers for box regression and centerness
        # operate on the output of `stem_box`.
        # See FCOS figure again; both stems are identical.
        #
        # Use `in_channels` and `stem_channels` for creating these layers, the
        # docstring above tells you what they mean. Initialize weights of each
        # conv layer from a normal distribution with mean = 0 and std dev = 0.01
        # and all biases with zero. Use conv stride = 1 and zero padding such
        # that size of input features remains same: remember we need predictions
        # at every location in feature map, we shouldn't "lose" any locations.
        ######################################################################
        # Fill these.
        stem_cls = []
        stem_box = []
        # Replace "pass" statement with your code

        prev_out = in_channels
        
        # 通过 stem_channels 控制 Stem 的深度和宽度，列表越长stem越深，列表中数值大小决定特征宽度
        # eg: stem_channels=[64, 64], 则有两层卷积
        for out_channels in stem_channels:
            #
            conv_cls = nn.Conv2d(prev_out, out_channels, kernel_size=3, stride=1, padding=1)
            nn.init.normal_(conv_cls.weight.data, mean=0, std=0.01)
            nn.init.constant_(conv_cls.bias, 0)
            stem_cls.append(conv_cls)
            stem_cls.append(nn.ReLU())
            
            #
            conv_box = nn.Conv2d(prev_out, out_channels, kernel_size=3, stride=1, padding=1)
            nn.init.normal_(conv_box.weight.data, 0, 0.01)
            nn.init.constant_(conv_box.bias.data, 0)
            stem_box.append(conv_box)
            stem_box.append(nn.ReLU())

            prev_out = out_channels

        # Wrap the layers defined by student into a `nn.Sequential` module:
        self.stem_cls = nn.Sequential(*stem_cls)
        self.stem_box = nn.Sequential(*stem_box)

        ######################################################################
        # TODO: Create THREE 3x3 conv layers for individually predicting three
        # things at every location of feature map:
        #     1. object class logits (`num_classes` outputs)
        #     2. box regression deltas (4 outputs: LTRB deltas from locations)
        #     3. centerness logits (1 output)
        #
        # Class probability and actual centerness are obtained by applying
        # sigmoid activation to these logits. However, DO NOT initialize those
        # modules here. This module should always output logits; PyTorch loss
        # functions have numerically stable implementations with logits. During
        # inference, logits are converted to probabilities by applying sigmoid,
        # BUT OUTSIDE this module.
        #
        ######################################################################

        # Replace these lines with your code, keep variable names unchanged.
        self.pred_cls = None  # Class prediction conv
        self.pred_box = None  # Box regression conv
        self.pred_ctr = None  # Centerness conv

        # Replace "pass" statement with your code
        self.pred_cls = nn.Conv2d(prev_out, num_classes, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.pred_cls.weight.data, mean=0, std=0.01)
        nn.init.constant_(self.pred_cls.bias.data, 0)
        
        self.pred_box = nn.Conv2d(prev_out, 4, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.pred_box.weight.data, mean=0, std=0.01)
        nn.init.constant_(self.pred_box.bias.data, 0)
        
        self.pred_ctr = nn.Conv2d(prev_out, 1, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.pred_ctr.weight.data, mean=0, std=0.01)
        nn.init.constant_(self.pred_ctr.bias.data, 0)
        
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # OVERRIDE: Use a negative bias in `pred_cls` to improve training
        # stability. Without this, the training will most likely diverge.
        # STUDENTS: You do not need to get into details of why this is needed.
        torch.nn.init.constant_(self.pred_cls.bias, -math.log(99))

    def forward(self, feats_per_fpn_level: TensorDict) -> List[TensorDict]:
        """
        Accept FPN feature maps and predict the desired outputs at every location
        (as described above). Format them such that channels are placed at the
        last dimension, and (H, W) are flattened (having channels at last is
        convenient for computing loss as well as perforning inference).

        Args:
            feats_per_fpn_level: Features from FPN, keys {"p3", "p4", "p5"}. Each
                tensor will have shape `(batch_size, fpn_channels, H, W)`. For an
                input (224, 224) image, H = W are (28, 14, 7) for (p3, p4, p5).
            feats_per_fpn_level:
                {
                    "p3": (batch_size, fpn_channels, H, W),
                    "p4": (batch_size, fpn_channels, H, W),
                    "p5": (batch_size, fpn_channels, H, W),
                }

        Returns:
            List of dictionaries, each having keys {"p3", "p4", "p5"}:
            1. Classification logits: `(batch_size, H * W, num_classes)`.
            2. Box regression deltas: `(batch_size, H * W, 4)`
            3. Centerness logits:     `(batch_size, H * W, 1)`
        """

        ######################################################################
        # TODO: Iterate over every FPN feature map and obtain predictions using
        # the layers defined above. Remember that prediction layers of box
        # regression and centerness will operate on output of `stem_box`,
        # and classification layer operates separately on `stem_cls`.
        #
        # CAUTION: The original FCOS model uses shared stem for centerness and
        # classification. Recent follow-up papers commonly place centerness and
        # box regression predictors with a shared stem, which we follow here.
        #
        # DO NOT apply sigmoid to classification and centerness logits.
        ######################################################################
        # Fill these with keys: {"p3", "p4", "p5"}, same as input dictionary.
        class_logits = {}
        boxreg_deltas = {}
        centerness_logits = {}

        # Replace "pass" statement with your code
        for key, value in feats_per_fpn_level.items():
            # classes predict
            class_logits[key] = self.pred_cls(self.stem_cls(value)) # (batch_size, num_classes, H, W)
            batch_size = class_logits[key].shape[0]
            num_classes = class_logits[key].shape[1]
            class_logits[key] = class_logits[key].reshape(batch_size, num_classes, -1) # (batch_size, num_classes, H*W)
            class_logits[key] = class_logits[key].permute(0, 2, 1) # (batch_size, H*W, num_classes)
            
            # box predict
            boxreg_deltas[key] = self.pred_box(self.stem_box(value))
            boxreg_deltas[key] = boxreg_deltas[key].reshape(batch_size, 4, -1)
            boxreg_deltas[key] = boxreg_deltas[key].permute(0, 2, 1)
            
            # center predict
            centerness_logits[key] = self.pred_ctr(self.stem_box(value))
            centerness_logits[key] = centerness_logits[key].reshape(batch_size, 1, -1)
            centerness_logits[key] = centerness_logits[key].permute(0, 2, 1)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        return [class_logits, boxreg_deltas, centerness_logits]


@torch.no_grad()
def fcos_match_locations_to_gt(
    locations_per_fpn_level: TensorDict,
    strides_per_fpn_level: Dict[str, int],
    gt_boxes: torch.Tensor,
) -> TensorDict:
    """
    Match centers of the locations of FPN feature with a set of GT bounding
    boxes of the input image. Since our model makes predictions at every FPN
    feature map location, we must supervise it with an appropriate GT box.
    There are multiple GT boxes in image, so FCOS has a set of heuristics to
    assign centers with GT, which we implement here.

    NOTE: This function is NOT BATCHED. Call separately for GT box batches.

    Args:
        locations_per_fpn_level: Centers at different levels of FPN (p3, p4, p5),
            that are already projected to absolute co-ordinates in input image
            dimension. Dictionary of three keys: (p3, p4, p5) giving tensors of
            shape `(H * W, 2)` where H = W is the size of feature map.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `common.py` for more details.
        gt_boxes: GT boxes of a single image, a batch of `(M, 5)` boxes with
            absolute co-ordinates and class ID `(x1, y1, x2, y2, C)`. In this
            codebase, this tensor is directly served by the dataloader.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(N, 5)` GT boxes, one for each center. They are
            one of M input boxes, or a dummy box called "background" that is
            `(-1, -1, -1, -1, -1)`. Background indicates that the center does
            not belong to any object.
    """

    """
        {
            "p3": None,
            "p4": None,
            "p5": None,
        }
    """
    matched_gt_boxes = {
        level_name: None for level_name in locations_per_fpn_level.keys()
    }

    """
        {
            "p3": (xc, yc) shape(H*W, 2),
            "p4": (xc, yc) shape(H2*W2, 2),
            "p5": (xc, yc) shape(H3*W3, 2),
        }
    """
    # Do this matching individually per FPN level., level_name: 'p3', 'p4', 'p5', centers: (784, 2), (196, 2), (49, 2)
    for level_name, centers in locations_per_fpn_level.items():

        # Get stride for this FPN level.  => (8, 16, 32)
        stride = strides_per_fpn_level[level_name]
        
        # shape(H*W, 2) => shape(H*W, 2, 1) => 沿着dim=1拆分成两个(H*W, 1)对应x坐标和y坐标
        x, y = centers.unsqueeze(dim=2).unbind(dim=1)
        # gt_boxes shape(M, 5) => shape(1, M, 4) => 4个(1, M) M表示GT框的数量
        x0, y0, x1, y1 = gt_boxes[:, :4].unsqueeze(dim=0).unbind(dim=2)
        # 计算中心点到GT框4个边界的距离 最后shape为(N, M, 4) N - 特征中心数量，M - GT框数量，4 -[Right, Bottom, Left, Top]的距离
        pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)

        # Pairwise distance between every feature center and GT box edges:
        # shape: (num_gt_boxes, num_centers_this_level, 4pairwise_dist)
        pairwise_dist = pairwise_dist.permute(1, 0, 2)

        # The original FCOS anchor matching rule: anchor point must be inside GT.
        """
            Rule 1: 位置是否在GT框内部
        """
        # 取 [Right, Bottom, Left, Top]距离的最小值 shape => (num_gt_boxes, num_centers_this_level)
        # 如果中心点在内部，则距离都为正数，如果在外部则至少会有一个距离为负数/0，此时得到一个布尔矩阵
        match_matrix = pairwise_dist.min(dim=2).values > 0

        """
            Rule 2: 多尺度匹配
        """
        # Multilevel anchor matching in FCOS: each anchor is only responsible
        # for certain scale range.
        # Decide upper and lower bounds of limiting targets.
        # 取 [Right, Bottom, Left, Top]距离的最大值 shape => (num_gt_boxes, num_centers_this_level)
        pairwise_dist = pairwise_dist.max(dim=2).values

        # 根据FPN层级设定多尺度匹配下限 p4, p5下限为 步长*4, p3下限为 0
        lower_bound = stride * 4 if level_name != "p3" else 0
        # 设定上限，p5为无穷不设上限，p3，p4上限为 步长*8
        upper_bound = stride * 8 if level_name != "p5" else float("inf")
        
        # 应用规则 - 只有当GT框的尺寸(pairwise_dist = pairwise_dist.max(dim=2).values)落在[lower_bound, upper_bound]范围内，
        # 并且特征中心点也在GT框内部(match_matrix 的前一部分结果)，才最终认为该位置和GT框匹配
        """
            在规则1上进一步应用规则2
            match_matrix = match_matrix & (pairwise_dist > lower_bound) & (pairwise_dist < upper_bound)
        """
        match_matrix &= (pairwise_dist > lower_bound) & (
            pairwise_dist < upper_bound
        )

        # Match the GT box with minimum area, if there are multiple GT matches.
        # 处理重叠框，选择面积最小的, gt_areas ：每个GT框的面积
        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (
            gt_boxes[:, 3] - gt_boxes[:, 1]
        )

        # Get matches and their labels using match quality matrix.
        match_matrix = match_matrix.to(torch.float32)
        # 用 GT框面积对match_matrix进行加权, .unsqueeze(-1)方便进行广播运算，1e8 - GT_面积 面积越大的GT框权重越小，取1e8为了确保面积小的框权重更大
        match_matrix *= 1e8 - gt_areas[:, None]

        # Find matched ground-truth instance per anchor (un-matched = -1).
        # 选择最佳匹配GT框
        # 对每个特征中心点(dim=0)在所有GT框中找到match_matrix值最大的那个GT框
        # torch.Size([784])， torch.Size([784])
        match_quality, matched_idxs = match_matrix.max(dim=0)
        # 如果最大匹配质量小于很小的阈值1e-5则把matched_idxs对应位置设为-1，表示特征中心点没有匹配到任何GT框，表示为背景
        matched_idxs[match_quality < 1e-5] = -1

        # Anchors with label 0 are treated as background.
        # 根据 matched_idxs 获取匹配到的 GT 框。
        # matched_idxs 中小于 0 的索引 (即 -1) clip 到 0。 防止索引越界，clip 到 0 后，
        # 对于未匹配到的位置，会错误地使用 gt_boxes[0] 的框信息， 但下一步会将其覆盖为背景框。
        # gt_boxes[...]: 使用 clip(clamp) 后的索引从 gt_boxes 中索引出对应的 GT 框。 形状为 (num_centers_this_level, 5)。
        # matched_idxs.clip(min=0) => [784, ] gt_boxes => [40, 5]，784个数挨个进行索引 torch.Size([784, 5])
        matched_boxes_this_level = gt_boxes[matched_idxs.clip(min=0)]
        # 将未匹配到的位置 (背景) 的 GT 框信息设置为背景框 (-1, -1, -1, -1, -1)
        matched_boxes_this_level[matched_idxs < 0, :] = -1

        matched_gt_boxes[level_name] = matched_boxes_this_level

    return matched_gt_boxes


def fcos_get_deltas_from_locations(
    locations: torch.Tensor, gt_boxes: torch.Tensor, stride: int
) -> torch.Tensor:
    """
    Compute distances from feature locations to GT box edges. These distances
    are called "deltas" - `(left, top, right, bottom)` or simply `LTRB`. The
    feature locations and GT boxes are given in absolute image co-ordinates.

    These deltas are used as targets for training FCOS to perform box regression
    and centerness regression. They must be "normalized" by the stride of FPN
    feature map (from which feature locations were computed, see the function
    `get_fpn_location_coords`). If GT boxes are "background", then deltas must
    be `(-1, -1, -1, -1)`.

    NOTE: This transformation function should not require GT class label. Your
    implementation must work for GT boxes being `(N, 4)` or `(N, 5)` tensors -
    without or with class labels respectively. You may assume that all the
    background boxes will be `(-1, -1, -1, -1)` or `(-1, -1, -1, -1, -1)`.

    Args:
        locations: Tensor of shape `(N, 2)` giving `(xc, yc)` feature locations. 
        gt_boxes: Tensor of shape `(N, 4 or 5)` giving GT boxes.
        stride: Stride of the FPN feature map.

    Returns:
        torch.Tensor
            Tensor of shape `(N, 4)` giving deltas from feature locations, that
            are normalized by feature stride.
    """
    ##########################################################################
    # TODO: Implement the logic to get deltas from feature locations.        #
    ##########################################################################
    # Set this to Tensor of shape (N, 4) giving deltas (left, top, right, bottom)
    # from the locations to GT box edges, normalized by FPN stride.
    deltas = None

    # Replace "pass" statement with your code
    deltas = torch.empty(size=(gt_boxes.shape[0], 4)).to(device=gt_boxes.device)
    
    deltas[:, 0] = locations[:, 0] - gt_boxes[:, 0] # l = (xc - x1)
    deltas[:, 1] = locations[:, 1] - gt_boxes[:, 1] # t = (yc - y1)
    deltas[:, 2] = gt_boxes[:, 2] - locations[:, 0] # r = (x2 - xc)
    deltas[:, 3] = gt_boxes[:, 3] - locations[:, 1] # b = (y2 - yc)
    deltas /= stride
    
    # assign delta of background to -1 | gt_boxes[:, :4] => (N, 4)后把4个坐标值加起来和-4比较
    # 一个 GT 框是背景框， 则坐标值加起来为-4 (-1, -1, -1, -1)
    deltas[gt_boxes[:, :4].sum(dim=1) == -4] = -1
    
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    # (N, 4)
    return deltas


def fcos_apply_deltas_to_locations(
    deltas: torch.Tensor, locations: torch.Tensor, stride: int
) -> torch.Tensor:
    """
    Implement the inverse of `fcos_get_deltas_from_locations` here:

    Given edge deltas (left, top, right, bottom) and feature locations of FPN, get
    the resulting bounding box co-ordinates by applying deltas on locations. This
    method is used for inference in FCOS: deltas are outputs from model, and
    applying them to anchors will give us final box predictions.

    Recall in above method, we were required to normalize the deltas by feature
    stride. Similarly, we have to un-normalize the input deltas with feature
    stride before applying them to locations, because the given input locations are
    already absolute co-ordinates in image dimensions.

    Args:
        deltas: Tensor of shape `(N, 4)` giving edge deltas to apply to locations.
        locations: Locations to apply deltas on. shape: `(N, 2)`
        stride: Stride of the FPN feature map.

    Returns:
        torch.Tensor
            Same shape as deltas and locations, giving co-ordinates of the
            resulting boxes `(x1, y1, x2, y2)`, absolute in image dimensions.
    """
    ##########################################################################
    # TODO: Implement the transformation logic to get boxes.                 #
    #                                                                        #
    # NOTE: The model predicted deltas MAY BE negative, which is not valid   #
    # for our use-case because the feature center must lie INSIDE the final  #
    # box. Make sure to clip them to zero.                                   #
    ##########################################################################
    # Replace "pass" statement with your code
    output_boxes = torch.empty_like(deltas)
    
    deltas = torch.clip(deltas, min=0)
    output_boxes[:, 0] = locations[:, 0] - deltas[:, 0] * stride # xc - l * stride
    output_boxes[:, 1] = locations[:, 1] - deltas[:, 1] * stride # yc - t * stride
    output_boxes[:, 2] = locations[:, 0] + deltas[:, 2] * stride # xc + r * stride
    output_boxes[:, 3] = locations[:, 1] + deltas[:, 3] * stride # yc + b * stride
    
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################

    return output_boxes


def fcos_make_centerness_targets(deltas: torch.Tensor):
    """
    Given LTRB deltas of GT boxes, compute GT targets for supervising the
    centerness regression predictor. See `fcos_get_deltas_from_locations` on
    how deltas are computed. If GT boxes are "background" => deltas are
    `(-1, -1, -1, -1)`, then centerness should be `-1`.

    For reference, centerness equation is available in FCOS paper
    https://arxiv.org/abs/1904.01355 (Equation 3).

    Args:
        deltas: Tensor of shape `(N, 4)` giving LTRB deltas for GT boxes.

    Returns:
        torch.Tensor
            Tensor of shape `(N, )` giving centerness regression targets.
    """
    ##########################################################################
    # TODO: Implement the centerness calculation logic.                      #
    ##########################################################################
    centerness = None
    # Replace "pass" statement with your code
    
    l_r = deltas[:, 0::2]
    t_b = deltas[:, 1::2]
    # shape = (N, )
    centerness = torch.sqrt(torch.min(l_r, dim=1).values * torch.min(t_b, dim=1).values 
                            / (torch.max(l_r, dim=1).values * torch.max(t_b, dim=1).values))
    
    centerness[deltas.sum(dim=1) == -4] = -1
    

    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################

    return centerness


class FCOS(nn.Module):
    """
    FCOS: Fully-Convolutional One-Stage Detector

    This class puts together everything you implemented so far. It contains a
    backbone with FPN, and prediction layers (head). It computes loss during
    training and predicts boxes during inference.
    """

    def __init__(
        self, num_classes: int, fpn_channels: int, stem_channels: List[int]
    ):
        super().__init__()
        self.num_classes = num_classes

        ######################################################################
        # TODO: Initialize backbone and prediction network using arguments.  #
        ######################################################################
        # Feel free to delete these two lines: (but keep variable names same)
        self.backbone = DetectorBackboneWithFPN(fpn_channels)
        self.pred_net = FCOSPredictionNetwork(num_classes, fpn_channels, stem_channels)
        # Replace "pass" statement with your code
        pass
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # Averaging factor for training loss; EMA of foreground locations.
        # STUDENTS: See its use in `forward` when you implement losses.
        self._normalizer = 150  # per image

    def forward(
        self,
        images: torch.Tensor,
        gt_boxes: Optional[torch.Tensor] = None,
        test_score_thresh: Optional[float] = None,
        test_nms_thresh: Optional[float] = None,
    ):
        """
        Args:
            images: Batch of images, tensors of shape `(B, C, H, W)`.
            gt_boxes: Batch of training boxes, tensors of shape `(B, N, 5)`.
                `gt_boxes[i, j] = (x1, y1, x2, y2, C)` gives information about
                the `j`th object in `images[i]`. The position of the top-left
                corner of the box is `(x1, y1)` and the position of bottom-right
                corner of the box is `(x2, x2)`. These coordinates are
                real-valued in `[H, W]`. `C` is an integer giving the category
                label for this bounding box. Not provided during inference.
            test_score_thresh: During inference, discard predictions with a
                confidence score less than this value. Ignored during training.
            test_nms_thresh: IoU threshold for NMS during inference. Ignored
                during training.

        Returns:
            Losses during training and predictions during inference.
        """

        ######################################################################
        # TODO: Process the image through backbone, FPN, and prediction head #
        # to obtain model predictions at every FPN location.                 #
        # Get dictionaries of keys {"p3", "p4", "p5"} giving predicted class #
        # logits, deltas, and centerness.                                    #
        ######################################################################
        # Feel free to delete this line: (but keep variable names same)
        pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits = None, None, None
        # Replace "pass" statement with your code
        fpn_feats = self.backbone(images)
        pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits = self.pred_net(fpn_feats)

        ######################################################################
        # TODO: Get absolute co-ordinates `(xc, yc)` for every location in
        # FPN levels.
        #
        # HINT: You have already implemented everything, just have to
        # call the functions properly.
        ######################################################################
        # Feel free to delete this line: (but keep variable names same)
        locations_per_fpn_level = None
        # Replace "pass" statement with your code
        fpn_feats_shapes = {
            "p3": fpn_feats["p3"].shape,
            "p4": fpn_feats["p4"].shape,
            "p5": fpn_feats["p5"].shape
        }
        
        locations_per_fpn_level = get_fpn_location_coords(fpn_feats_shapes,
                                                        self.backbone.fpn_strides, device=images.device)
        
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        if not self.training:
            # During inference, just go to this method and skip rest of the
            # forward pass.
            # fmt: off
            return self.inference(
                images, locations_per_fpn_level,
                pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits,
                test_score_thresh=test_score_thresh,
                test_nms_thresh=test_nms_thresh,
            )
            # fmt: on

        ######################################################################
        # TODO: Assign ground-truth boxes to feature locations. We have this
        # implemented in a `fcos_match_locations_to_gt`. This operation is NOT
        # batched so call it separately per GT boxes in batch.
        ######################################################################
        # List of dictionaries with keys {"p3", "p4", "p5"} giving matched
        # boxes for locations per FPN level, per image. Fill this list:
        matched_gt_boxes = []
        # Replace "pass" statement with your code
        pass

        # Calculate GT deltas for these matched boxes. Similar structure
        # as `matched_gt_boxes` above. Fill this list:
        matched_gt_deltas = []
        # Replace "pass" statement with your code
        pass
        
        # GT框的匹配和delta计算是非批次化的操作，需要对batch中的每个图像单独处理
        for batch in range(gt_boxes.shape[0]):
            # 为每个图像初始化 `匹配的GT框` 和 `GT deltas`
            matched_gt_boxes_batch = {}
            matched_gt_deltas_batch = {}
            
            # 将GT框匹配到特征位置
            matched_boxes_per_fpn_level = fcos_match_locations_to_gt(locations_per_fpn_level,
                                                                     self.backbone.fpn_strides, gt_boxes[batch])
            
            # 将当前图像的GT框匹配到所有FPN层级的位置坐标
            for level_name, matched_boxes in matched_boxes_per_fpn_level.items():
                matched_gt_boxes_batch[level_name] = matched_boxes
                matched_gt_deltas_batch[level_name] = fcos_get_deltas_from_locations(locations_per_fpn_level[level_name], 
                                                                                    matched_boxes, self.backbone.fpn_strides[level_name])
            matched_gt_boxes.append(matched_gt_boxes_batch)
            matched_gt_deltas.append(matched_gt_deltas_batch)
            
            
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # Collate lists of dictionaries, to dictionaries of batched tensors.
        # These are dictionaries with keys {"p3", "p4", "p5"} and values as
        # tensors of shape (batch_size, locations_per_fpn_level, 5 or 4)
        matched_gt_boxes = default_collate(matched_gt_boxes)
        matched_gt_deltas = default_collate(matched_gt_deltas)

        # Combine predictions and GT from across all FPN levels.
        # shape: (batch_size, num_locations_across_fpn_levels, ...)
        matched_gt_boxes = self._cat_across_fpn_levels(matched_gt_boxes)
        matched_gt_deltas = self._cat_across_fpn_levels(matched_gt_deltas)
        pred_cls_logits = self._cat_across_fpn_levels(pred_cls_logits)
        pred_boxreg_deltas = self._cat_across_fpn_levels(pred_boxreg_deltas)
        pred_ctr_logits = self._cat_across_fpn_levels(pred_ctr_logits)

        # Perform EMA update of normalizer by number of positive locations.
        num_pos_locations = (matched_gt_boxes[:, :, 4] != -1).sum()
        pos_loc_per_image = num_pos_locations.item() / images.shape[0]
        self._normalizer = 0.9 * self._normalizer + 0.1 * pos_loc_per_image

        #######################################################################
        # TODO: Calculate losses per location for classification, box reg and
        # centerness. Remember to set box/centerness losses for "background"
        # positions to zero.
        ######################################################################
        # Feel free to delete this line: (but keep variable names same)
        loss_cls, loss_box, loss_ctr = None, None, None

        # Replace "pass" statement with your code
        B, L, _ = matched_gt_boxes.shape
        #  GT 框的类别标签转换为 one-hot 编码形式，并去掉背景类别
        matched_gt_classes = F.one_hot((matched_gt_boxes[:, :, 4] + 1).to(dtype=int), num_classes=21)[:, :, 1:].to(device=matched_gt_boxes.device, dtype=matched_gt_boxes.dtype)
        
        # 计算分类损失
        loss_cls = sigmoid_focal_loss(pred_cls_logits, matched_gt_classes)
        # 计算框回归损失
        loss_box = 0.25 * F.l1_loss(pred_boxreg_deltas, matched_gt_deltas, reduction="none")
        # 背景框设置为0
        loss_box[matched_gt_deltas < 0] *= 0.0
        
        # 初始化 (B*L,) 的全零 Tensor
        matched_gt_ctr = torch.zeros(B*L).to(device=matched_gt_boxes.device, dtype=matched_gt_boxes.dtype)
        # flatten
        pred_ctr_logits = pred_ctr_logits.view(-1) 
        matched_gt_deltas = matched_gt_deltas.view(-1, 4)
        
        # 计算GT中心度目标值
        matched_gt_ctr = fcos_make_centerness_targets(matched_gt_deltas)
        # 中心度损失
        loss_ctr = F.binary_cross_entropy_with_logits(pred_ctr_logits, matched_gt_ctr, reduction="none")
        loss_ctr[matched_gt_ctr < 0] *= 0.0
        
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################
        # Sum all locations and average by the EMA of foreground locations.
        # In training code, we simply add these three and call `.backward()`
        return {
            "loss_cls": loss_cls.sum() / (self._normalizer * images.shape[0]),
            "loss_box": loss_box.sum() / (self._normalizer * images.shape[0]),
            "loss_ctr": loss_ctr.sum() / (self._normalizer * images.shape[0]),
        }

    @staticmethod
    def _cat_across_fpn_levels(
        dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int = 1
    ):
        """
        Convert a dict of tensors across FPN levels {"p3", "p4", "p5"} to a
        single tensor. Values could be anything - batches of image features,
        GT targets, etc.
        """
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)

    def inference(
        self,
        images: torch.Tensor,
        locations_per_fpn_level: Dict[str, torch.Tensor],
        pred_cls_logits: Dict[str, torch.Tensor],
        pred_boxreg_deltas: Dict[str, torch.Tensor],
        pred_ctr_logits: Dict[str, torch.Tensor],
        test_score_thresh: float = 0.3,
        test_nms_thresh: float = 0.5,
    ):
        """
        Run inference on a single input image (batch size = 1). Other input
        arguments are same as those computed in `forward` method. This method
        should not be called from anywhere except from inside `forward`.

        Returns:
            Three tensors:
                - pred_boxes: Tensor of shape `(N, 4)` giving *absolute* XYXY
                  co-ordinates of predicted boxes.

                - pred_classes: Tensor of shape `(N, )` giving predicted class
                  labels for these boxes (one of `num_classes` labels). Make
                  sure there are no background predictions (-1).

                - pred_scores: Tensor of shape `(N, )` giving confidence scores
                  for predictions: these values are `sqrt(class_prob * ctrness)`
                  where class_prob and ctrness are obtained by applying sigmoid
                  to corresponding logits.
        """

        # Gather scores and boxes from all FPN levels in this list. Once
        # gathered, we will perform NMS to filter highly overlapping predictions.
        pred_boxes_all_levels = []
        pred_classes_all_levels = []
        pred_scores_all_levels = []

        for level_name in locations_per_fpn_level.keys():

            # Get locations and predictions from a single level.
            # We index predictions by `[0]` to remove batch dimension. 去除批次维度，推理时batch size = 1
            level_locations = locations_per_fpn_level[level_name]
            level_cls_logits = pred_cls_logits[level_name][0]
            level_deltas = pred_boxreg_deltas[level_name][0]
            level_ctr_logits = pred_ctr_logits[level_name][0]

            ##################################################################
            # TODO: FCOS uses the geometric mean of class probability and
            # centerness as the final confidence score. This helps in getting
            # rid of excessive amount of boxes far away from object centers.
            # Compute this value here (recall sigmoid(logits) = probabilities)
            #
            # Then perform the following steps in order:
            #   1. Get the most confidently predicted class and its score for
            #      every box. Use level_pred_scores: (N, num_classes) => (N, )
            #   2. Only retain prediction that have a confidence score higher
            #      than provided threshold in arguments.
            #   3. Obtain predicted boxes using predicted deltas and locations
            #   4. Clip XYXY box-cordinates that go beyond thr height and
            #      and width of input image.
            ##################################################################
            # Feel free to delete this line: (but keep variable names same)
            level_pred_boxes, level_pred_classes, level_pred_scores = (
                None,
                None,
                None,  # Need tensors of shape: (N, 4) (N, ) (N, )
            )

            # Compute geometric mean of class logits and centerness: 计算置信度得分
            level_pred_scores = torch.sqrt(
                level_cls_logits.sigmoid_() * level_ctr_logits.sigmoid_()
            )
            # Step 1:
            # Replace "pass" statement with your code
            # 获取置信度最高的类别和得分
            level_pred_scores, classes = level_pred_scores.max(dim=1)

            # Step 2:
            # Replace "pass" statement with your code
            # 得分阈值过滤
            keep = level_pred_scores > test_score_thresh
            level_pred_classes = classes[keep]
            level_pred_scores = level_pred_scores[keep]
            
            # Step 3: 框解码
            # Replace "pass" statement with your code
            # 把预测的deltas应用到位置坐标上
            level_pred_boxes = fcos_apply_deltas_to_locations(level_deltas, level_locations, 
                                stride=self.backbone.fpn_strides[level_name])
            level_pred_boxes = level_pred_boxes[keep]
            
            # no background 移除背景框的类别预测
            not_background = (level_deltas[keep].sum(dim=1) != -4)
            level_pred_boxes = level_pred_boxes[not_background]
            level_pred_classes = level_pred_classes[not_background]
            level_pred_scores = level_pred_scores[not_background]

            # Step 4: Use `images` to get (height, width) for clipping.
            # Replace "pass" statement with your code
            # 框坐标裁剪
            level_pred_boxes[:, 0] = level_pred_boxes[:, 0].clip(min=0)
            level_pred_boxes[:, 1] = level_pred_boxes[:, 1].clip(min=0)
            level_pred_boxes[:, 2] = level_pred_boxes[:, 2].clip(max=images.shape[2])
            level_pred_boxes[:, 3] = level_pred_boxes[:, 3].clip(max=images.shape[3])
            ##################################################################
            #                          END OF YOUR CODE                      #
            ##################################################################

            pred_boxes_all_levels.append(level_pred_boxes)
            pred_classes_all_levels.append(level_pred_classes)
            pred_scores_all_levels.append(level_pred_scores)

        ######################################################################
        # Combine predictions from all levels and perform NMS.
        pred_boxes_all_levels = torch.cat(pred_boxes_all_levels)
        pred_classes_all_levels = torch.cat(pred_classes_all_levels)
        pred_scores_all_levels = torch.cat(pred_scores_all_levels)

        # STUDENTS: This function depends on your implementation of NMS.
        keep = class_spec_nms(
            pred_boxes_all_levels,
            pred_scores_all_levels,
            pred_classes_all_levels,
            iou_threshold=test_nms_thresh,
        )
        pred_boxes_all_levels = pred_boxes_all_levels[keep]
        pred_classes_all_levels = pred_classes_all_levels[keep]
        pred_scores_all_levels = pred_scores_all_levels[keep]
        return (
            pred_boxes_all_levels,
            pred_classes_all_levels,
            pred_scores_all_levels,
        )
