import torch
from torchvision.ops.boxes import box_area


def bbox_cxcywh_to_xyxy(x: torch.Tensor):
    """convert a bounding box from format cxcywh to xyxy

    :param x: bounding box tensor with cxcywh format
    :return: a tensor with xyxy format
    """
    cx, cy, w, h = x.unbind(-1)
    b = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(b, dim=-1)


def bbox_xyxy_to_cxcywh(x: torch.Tensor):
    """convert a bounding box from format xyxy to cxcywh

    :param x: bounding box tensor with xyxy format
    :return: a tensor with cxcywh format
    """
    xmin, ymin, xmax, ymax = x.unbind(-1)
    b = [(xmin + xmax) / 2, (ymin + ymax) / 2, (xmax - xmin), (ymax - ymin)]
    return torch.stack(b, dim=-1)


def bbox_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """calculate IoU from two bounding boxes

    :param boxes1: bounding box tensor 1
    :param boxes2: bounding box tensor 2
    :return: IoU value and Union Value
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    top_left = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    bottom_right = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (bottom_right - top_left).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter
    iou = inter / union

    return iou, union


def generalized_bbox_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """generalized IoU.

    :param boxes1: bounding box tensor 1 in xyxy format
    :param boxes2: bounding box tensor 2 in xyxy format
    :return: an [N, M] pairwise matrix, where N = len(boxes1), M = len(boxes2)
    """
    assert torch.BoolTensor(boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert torch.BoolTensor(boxes1[:, 2:] >= boxes1[:, :2]).all()
    iou, union = bbox_iou(boxes1, boxes2)

    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (bottom_right - top_left).clamp(min=0)  # [N, M, 2]
    area = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    return iou - (area - union) / area
