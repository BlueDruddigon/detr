from typing import Dict, List

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from utils.bbox_ops import generalized_bbox_iou, bbox_cxcywh_to_xyxy


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the `no-object`. Because of this, in general,
    there are more predictions than the targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are unmatched (and thus treated as `non-object`)
    """
    def __init__(self, cost_class: float = 1., cost_bbox: float = 1., cost_giou: float = 1.) -> None:
        """Creates the matcher

        Params:
            cost_class: The relative weight of the classification error in the matching cost
            cost_bbox: The relative weight of the bounding box L1 error in the matching cost
            cost_giou: The relative weight of the bounding box GIoU error in the matching cost

        """
        super(HungarianMatcher, self).__init__()
        
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        """
        Args:
            outputs (Dict): contains prediction set of objects
                "pred_logits": Tensor of [batch_size, num_queries, num_classes] with the classification logits
                "pred_boxes": Tensor of [batch_size, num_queries, 4] with the predicted box coordinates
            targets (List[Dict]): contains list of targets, where each element is a dict contains:
                "labels": Tensor of [num_target_boxes]
                    (num_target_boxes is number of ground truth objects in targets) contains the class label
                "boxes": Tensor of [num_target_boxes, 4] contains the target box coordinates

        Returns:
            A list of (index_i, index_j) tuples, where:
                index_i: the indices of the selected predictions
                index_j: the indices of the corresponding selected targets
            and, `len(index_i) = len(index_j) = min(num_queries, num_target_boxes)`
        """
        batch_size, num_queries, _ = outputs['pred_logits'].shape
        
        # We flatten to compute the cost matrices in a batch
        out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)  # (batch_size * num_queries, num_classes)
        out_bbox = outputs['pred_boxes'].flatten(0, 1)  # (batch_size * num_queries, 4)
        
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([t['labels'] for t in targets])
        tgt_bbox = torch.cat([t['boxes'] for t in targets])
        
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, and it can be ommitted.
        cost_class = 1 - out_prob[:, tgt_ids]
        
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # Compute GIoU cost between boxes
        cost_giou = -generalized_bbox_iou(bbox_cxcywh_to_xyxy(out_bbox), bbox_cxcywh_to_xyxy(tgt_bbox))
        
        # Final cost matrix
        C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
        C = C.view(batch_size, num_queries, -1)
        
        sizes = [len(t['boxes']) for t in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
