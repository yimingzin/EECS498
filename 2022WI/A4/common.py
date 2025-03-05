"""
This module contains classes and functions that are common across both, one-stage
and two-stage detector implementations. You have to implement some parts here -
walk through the notebooks and you will find instructions on *when* to implement
*what* in this module.
"""
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import feature_extraction


def hello_common():
    print("Hello from common.py!")


class DetectorBackboneWithFPN(nn.Module):
    """
    Detection backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (out_channels, H /  8, W /  8)      stride =  8
        - level p4: (out_channels, H / 16, W / 16)      stride = 16
        - level p5: (out_channels, H / 32, W / 32)      stride = 32

    NOTE: We could use any convolutional network architecture that progressively
    downsamples the input image and couple it with FPN. We use a small enough
    backbone that can work with Colab GPU and get decent enough performance.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights.
        _cnn = models.regnet_x_400mf(pretrained=True)

        # Torchvision models only return features from the last level. Detector
        # backbones (with FPN) require intermediate features of different scales.
        # So we wrap the ConvNet with torchvision's feature extractor. Here we
        # will get output features with names (c3, c4, c5) with same stride as
        # (p3, p4, p5) described above.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )

        # Pass a dummy batch of input images to infer shapes of (c3, c4, c5).
        # Features are a dictionary with keys as defined above. Values are
        # batches of tensors in NCHW format, that give intermediate features
        # from the backbone network.
        """dummy_out_shapes: List of Tuple
            [
                ('c3', torch.Size([2, 64, 28, 28])),
                ('c4', torch.Size([2, 160, 14, 14])),
                ('c5', torch.Size([2, 400, 7, 7]))
            ]
        """
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        print("For dummy input images with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in dummy_out_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")

        ######################################################################
        # TODO: Initialize additional Conv layers for FPN.                   #
        #                                                                    #
        # Create THREE "lateral" 1x1 conv layers to transform (c3, c4, c5)   #
        # such that they all end up with the same `out_channels`.            #
        # Then create THREE "output" 3x3 conv layers to transform the merged #
        # FPN features to output (p3, p4, p5) features.                      #
        # All conv layers must have stride=1 and padding such that features  #
        # do not get downsampled due to 3x3 convs.                           #
        #                                                                    #
        # HINT: You have to use `dummy_out_shapes` defined above to decide   #
        # the input/output channels of these layers.                         #
        ######################################################################
        # This behaves like a Python dict, but makes PyTorch understand that
        # there are trainable weights inside it.
        # Add THREE lateral 1x1 conv and THREE output 3x3 conv layers.
        self.fpn_params = nn.ModuleDict()

        # Replace "pass" statement with your code
        self.fpn_params = nn.ModuleDict({
            # 来自RegNet的c3, c4, c5可能具有不同的通道数，需要将他们统一为相同通道数
            "m3": nn.Conv2d(dummy_out_shapes[0][1][1], out_channels, kernel_size=1, stride=1, padding=0),
            "m4": nn.Conv2d(dummy_out_shapes[1][1][1], out_channels, kernel_size=1, stride=1, padding=0),
            "m5": nn.Conv2d(dummy_out_shapes[2][1][1], out_channels, kernel_size=1, stride=1, padding=0),
            
            "p3": nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            "p4": nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            "p5": nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        })
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size. You may access these values freely
        to implement your logic in FCOS / Faster R-CNN.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):

        # Multi-scale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}
        ######################################################################
        # TODO: Fill output FPN features (p3, p4, p5) using RegNet features  #
        # (c3, c4, c5) and FPN conv layers created above.                    #
        # HINT: Use `F.interpolate` to upsample FPN features.                #
        ######################################################################

        # Replace "pass" statement with your code
        m3 = self.fpn_params["m3"](backbone_feats["c3"])
        m4 = self.fpn_params["m4"](backbone_feats["c4"])
        m5 = self.fpn_params["m5"](backbone_feats["c5"])
        
        """ 自顶向下，上采样相加融合，
            ('c5', torch.Size([2, 400, 7, 7]))  -> ('m5', torch.Size([2, out_channel, 7, 7])) ->  ('m5_upsample', torch.Size([2, out_channel, 14, 14]))
            ('c4', torch.Size([2, 160, 14, 14])) -> ('m4', torch.Size([2, out_channel, 14, 14])) + m5_upsample -> ('m4_upsample', torch.Size([2, out_channel, 28, 28])   
            ('c3', torch.Size([2, 64, 28, 28])) -> ('m3', torch.Size([2, out_channel, 28, 28]) + m4_usample
        """
        
        m5_upsample = F.interpolate(m5, size=(m4.shape[2], m4.shape[3]), mode="nearest")
        m4 = m4 + m5_upsample
        
        m4_upsample = F.interpolate(m4, size=(m3.shape[2], m3.shape[3]), mode="nearest")
        m3 = m3 + m4_upsample
        
        # p3, p4, p5
        fpn_feats["p3"] = self.fpn_params["p3"](m3)
        fpn_feats["p4"] = self.fpn_params["p4"](m4)
        fpn_feats["p5"] = self.fpn_params["p5"](m5)
        
        
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        return fpn_feats


def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Map every location in FPN feature map to a point on the image. This point
    represents the center of the receptive field of this location. We need to
    do this for having a uniform co-ordinate representation of all the locations
    across FPN levels, and GT boxes.

    Args:
        shape_per_fpn_level: Shape of the FPN feature level, dictionary of keys
            {"p3", "p4", "p5"} and feature shapes `(B, C, H, W)` as values.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `backbone.py` for more details.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(H * W, 2)` giving `(xc, yc)` co-ordinates of the
            centers of receptive fields of the FPN locations, on input image.
    """

    # Set these to `(N, 2)` Tensors giving absolute location co-ordinates.
    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        ######################################################################
        # TODO: Implement logic to get location co-ordinates below.          #
        ######################################################################
        # Replace "pass" statement with your code
        _, _, H, W = feat_shape # 解包特征图特征，获取高度 h 和宽度w
        
        # 1.生成特征图上的网格索引 (i, j)
        rows = torch.arange(H, dtype=dtype, device=device)
        cols = torch.arange(W, dtype=dtype, device=device)
        grid_y, grid_x = torch.meshgrid(rows, cols, indexing='ij') #使用indexing = 'ij' 确保先y后x
        
        # 2. 计算每个位置的 (xc, yc) 坐标
        xc = level_stride * (grid_x + 0.5) # 公式：stride * (j + 0.5)， grid_x 对应列索引 j
        yc = level_stride * (grid_y + 0.5) # 公式：stride * (i + 0.5)， grid_y 对应行索引 i
        
        # 3. 将 (xc, yc)坐标堆叠成 (H*W, 2)
        location_coords_level = torch.stack([xc, yc], dim=-1) # shape (H, W, 2)
        # 方便遍历和后续处理
        location_coords[level_name] = location_coords_level.reshape(-1, 2) # shape (H * W, 2)
        
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################
    return location_coords


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    """
    Non-maximum suppression removes overlapping bounding boxes.

    Args:
        boxes: Tensor of shape (N, 4) giving top-left and bottom-right coordinates
            of the bounding boxes to perform NMS on.
        scores: Tensor of shpe (N, ) giving scores for each of the boxes.
        iou_threshold: Discard all overlapping boxes with IoU > iou_threshold

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """

    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long)

    keep = None
    #############################################################################
    # TODO: Implement non-maximum suppression which iterates the following:     #
    #       1. Select the highest-scoring box among the remaining ones,         #
    #          which has not been chosen in this step before                    #
    #       2. Eliminate boxes with IoU > threshold                             #
    #       3. If any boxes remain, GOTO 1                                      #
    #       Your implementation should not depend on a specific device type;    #
    #       you can use the device of the input if necessary.                   #
    # HINT: You can refer to the torchvision library code:                      #
    # github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/nms_kernel.cpp
    #############################################################################
    # Replace "pass" statement with your code
    keep = []
    x1, y1, x2, y2 = boxes[:, :4].unbind(dim=1)
    area = torch.mul(x2 - x1, y2 - y1)
    
    # 对得分进行排序，并获取排序后的索引(升序)
    _, index = scores.sort(0)
    
    count = 0
    # 只要index中还有未处理的框, 就继续循环 - index.numel() 返回 `index` 中 Tensor 的总数
    while index.numel() > 0:
        # 获取当前index中最后一个索引 (得分最高)
        largest_idx = index[-1]
        keep.append(largest_idx)
        count += 1
        # 移除得分最高的
        index = index[:-1]
        
        if index.size(0) == 0:
            break
        
        # get the x1,y1,x2,y2 of all the remaining boxes, and clamp them so that
        # we get the coord of intersection of boxes and highest-scoring box
        # 计算最高分框和每个剩余框的交集矩形坐标，如果没有交集就会变为负数,需要处理
        x1_inter = torch.index_select(x1, 0, index).clamp(min=x1[largest_idx])
        y1_inter = torch.index_select(y1, 0, index).clamp(min=y1[largest_idx])
        x2_inter = torch.index_select(x2, 0, index).clamp(max=x2[largest_idx])
        y2_inter = torch.index_select(y2, 0, index).clamp(max=y2[largest_idx])
        
        # 计算面积
        W_inter = (x2_inter - x1_inter).clamp(min=0.0)
        H_inter = (y2_inter - y1_inter).clamp(min=0.0)
        inter_area = W_inter * H_inter
        
        # 计算并集区域的面积 area - 并集区域面积 | union - 框A面积 + 框B面积 - 交集面积
        areas = torch.index_select(area, 0, index)
        union_area = (areas - inter_area) + area[largest_idx]
        
        # IoU 判断哪些元素小于等于iou_threshold
        IoU = inter_area / union_area
        index = index[IoU.le(iou_threshold)]
    
    keep = torch.Tensor(keep).to(device=scores.device).long()
        
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return keep


def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep
