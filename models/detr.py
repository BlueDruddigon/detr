from typing import List, Dict, Tuple, Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F

from utils.bbox_ops import bbox_cxcywh_to_xyxy, generalized_bbox_iou
from utils.misc import (
  NestedTensor, nested_tensor_from_tensor_list, get_world_size, is_dist_available_and_initialized, accuracy
)
from .backbone import build_backbone
from .common import MLP
from .matcher import build_matcher
from .transformer import build_transformer


class DETR(nn.Module):
    """This is the DETR module that performs object detection"""
    def __init__(
      self,
      backbone: nn.Module,
      transformer: nn.Module,
      num_classes: int,
      num_queries: int,
      aux_loss: bool = False
    ) -> None:
        """initializes the model

        :param backbone: nn.Module the backbone to be used.
        :param transformer: nn.Module the transformer architecture.
        :param num_classes: number of object classes.
        :param num_queries: number of object queries. this is the maximal number of objects DETR can detect in a single
                            image. FOR COCO, we recommend 100 queries.
        :param aux_loss: if true, auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super(DETR, self).__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        
        self.transformer = transformer
        self.backbone = backbone
        self.aux_loss = aux_loss
        
        embed_dim = transformer.embed_dim
        self.class_embed = nn.Linear(embed_dim, num_classes + 1)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4)
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, embed_dim, kernel_size=1)
    
    def forward(self, samples: NestedTensor):
        """the forward pass

        :param samples: a NestedTensor, which contains:
            - samples.tensor: batched images, of shape (batch_size, 3, H, W)
            - samples.mask: a binary mask of shape (batch_size, H, W), contains 1 on padded pixels
        :return: a dict with elements:
            - "pred_logits": the classification probabilities (including `no-object`) for all queries.
                shape = (batch_size, num_queries, num_classes + 1)
            - "pred_boxes": the normalized bounding boxes coordinates for all queries, represented as
                (center_x, center_y, height, width). The values are normalized in [0, 1], relative to the size of each
                image (disregarding possible padding).
            - "aux_outputs": Optional, only returned when auxiliary losses are activated. It is a list of dictionaries
                containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        # Feature maps and Positional Encoding
        features, pos = self.backbone(samples)
        
        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class: torch.Tensor, outputs_coord: torch.Tensor):
        """Torchscript handler, since Torchscript does not allow dictionary with non-homogeneous values, such as a
        dict with both a Tensor and a list.
        """
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1. Compute the hungarian assignment between ground truth boxes and predictions
        2. Supervise each pair of matched ground truth / prediction (both class and box)
    """
    def __init__(
      self, num_classes: int, matcher: nn.Module, weight_dict: Dict[str, float], eos_coef: float, losses: List[str]
    ) -> None:
        """Creates the Criterion

        :param num_classes: number of object categories, omitting the special `no-object` class.
        :param matcher: nn.Module to compute the matching between the outputs and targets
        :param weight_dict: dict containing as a key the names of the losses and as values their relative weight.
        :param eos_coef: relative classification weight applied to the `no-object` class.
        :param losses: list of all the losses to be applied.
        """
        super(SetCriterion, self).__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
    
    def loss_labels(
      self,
      outputs: Dict[str, torch.Tensor],
      targets: List[Dict[str, torch.Tensor]],
      indices: Tuple[torch.Tensor, torch.Tensor],
      num_boxes: float,
      log: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Classification loss (NLL)

        :param outputs: Dict[str, torch.Tensor] - outputs of DETR class.
        :param targets: List[Dict[str, torch.Tensor]] - targets got in DataLoader.
        :param indices: Tuple[torch.Tensor, torch.Tensor] - outputs of Matcher.
        :param num_boxes: float - number of boxes.
        :param log: boolean - if True, log the loss
        :return: a dict contains loss value of classification.
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        
        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses
    
    @torch.no_grad()
    def loss_cardinality(
      self,
      outputs: Dict[str, torch.Tensor],
      targets: List[Dict[str, torch.Tensor]],
      indices: Tuple[torch.Tensor, torch.Tensor],
      num_boxes: float,
    ) -> Dict[str, torch.Tensor]:
        """Computes the cardinality error, i.e., the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients

        :param outputs: Dict[str, torch.Tensor] - outputs of DETR class.
        :param targets: List[Dict[str, torch.Tensor]] - targets got in DataLoader.
                        targets dicts must contain the "labels" key containing a tensor with dim (num_target_boxes)
        :param indices: Tuple[torch.Tensor, torch.Tensor] - outputs of Matcher.
        :param num_boxes: float - number of boxes.
        :return: a dict contains the cardinality error
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_error = F.l1_loss(card_pred.float(), tgt_lengths.float())
        return {'cardinality_error': card_error}
    
    def loss_boxes(
      self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]],
      indices: Tuple[torch.Tensor, torch.Tensor], num_boxes: float
    ) -> Dict[str, torch.Tensor]:
        """Computes the loss related to the bounding boxes, the L1 regression loss and the GIoU loss.

        :param outputs: Dict[str, torch.Tensor] - outputs of DETR class.
        :param targets: List[Dict[str, torch.Tensor]] - targets got in DataLoader.
                        targets dicts must contain the "boxes" key containing a tensor with dim (num_target_boxes, 4)
                        target boxes in cxcywh format.
        :param indices: Tuple[torch.Tensor, torch.Tensor] - outputs of Matcher.
        :param num_boxes: float - number of boxes.
        :return: a dict contains combination bounding box losses.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_giou = 1 - torch.diag(
          generalized_bbox_iou(bbox_cxcywh_to_xyxy(src_boxes), bbox_cxcywh_to_xyxy(target_boxes))
        )
        return {
          'loss_bbox': loss_bbox.sum() / num_boxes,
          'loss_giou': loss_giou.sum() / num_boxes,
        }
    
    @staticmethod
    def _get_src_permutation_idx(indices: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Permute predictions following indices

        :param indices: Tuple[torch.Tensor, torch.Tensor] - outputs of Matcher.
        :return: Tuple[torch.Tensor, torch.Tensor] - the batch idx and src idx
        """
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    @staticmethod
    def _get_tgt_permutation_idx(indices: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Permute targets following indices

        :param indices: Tuple[torch.Tensor, torch.Tensor] - outputs of Matcher.
        :return: Tuple[torch.Tensor, torch.Tensor] - the batch idx and src idx
        """
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
    def get_loss(
      self, loss: str, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]],
      indices: Tuple[torch.Tensor, torch.Tensor], num_boxes: float, **kwargs: Any
    ):
        """Get corresponding loss values from name

        :param loss: str - the name of loss function
        :param outputs: Dict[str, torch.Tensor] - outputs of DETR class.
        :param targets: List[Dict[str, torch.Tensor]] - targets got in DataLoader.
        :param indices: Tuple[torch.Tensor, torch.Tensor] - outputs of Matcher.
        :param num_boxes: float - number of boxes.
        :param kwargs: Any - other arguments
        :return: Dict[str, torch.Tensor] - dict contains loss name as key and loss output as value
        """
        loss_map = {'labels': self.loss_labels, 'cardinality': self.loss_cardinality, 'boxes': self.loss_boxes}
        assert loss in loss_map, f'Do you really want to compute {loss} loss ?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str,
                                                                           torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """This performs the loss computation

        :param outputs: Dict[str, torch.Tensor] - outputs of DETR class.
        :param targets: List[Dict[str, torch.Tensor]] - targets got in DataLoader.
        :return: Dict[str, torch.Tensor] - Total losses updated in dict
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        
        # Compute the average number of target boxes across all nodes, for normalization purpose
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            dist.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses |= self.get_loss(loss, outputs, targets, indices, num_boxes)
        
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {f'{k}_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        return losses


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the COCO API"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """Performs the computation

        :param outputs: The raw outputs of the model
        :param target_sizes: tensor with dimension (batch_size, 2) contain the size of each image in the batch.
            For evaluation, this must be the original image size (before any data augmentation).
            For visualization, this should be the image size after data augment, but before padding.
        :return: a list of dictionaries contains scores, labels and boxes.
        """
        output_logits, output_bbox = outputs['pred_logits'], outputs['pred_boxes']
        
        assert len(output_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        
        prob = F.softmax(output_logits, -1)
        scores, labels = prob[..., :-1].max(-1)
        
        # convert to xyxy format
        boxes = bbox_cxcywh_to_xyxy(output_bbox)
        # and from relative [0, 1] to absolute coords
        img_height, img_width = target_sizes.unbind(1)
        scale_factor = torch.stack([img_width, img_height, img_width, img_height], dim=1)
        boxes = boxes * scale_factor[:, None, :]
        
        return [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]


def build(args):
    num_classes = 20 if args.dataset_file != 'coco' else 91
    device = torch.device(args.device)
    
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    
    model = DETR(
      backbone,
      transformer,
      num_classes=args.num_classes,
      num_queries=args.num_queries,
      aux_loss=args.aux_loss,
    )
    
    matcher = build_matcher(args)
    weight_dict = {
      'loss_ce': 1,
      'loss_bbox': args.bbox_loss_coef,
      'loss_giou': args.giou_loss_coef,
    }
    
    # Auxiliary losses
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict |= {f'{k}_{i}': v for k, v in weight_dict.items()}
        weight_dict |= aux_weight_dict
    
    losses = ['labels', 'boxes', 'cardinality']
    
    criterion = SetCriterion(
      num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses
    )
    criterion.to(device)
    post_processors = {'bbox': PostProcess()}
    
    return model, criterion, post_processors
