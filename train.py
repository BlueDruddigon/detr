import argparse

import torch

from models import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Training
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr-backbone', default=1e-5, type=float)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr-drop', default=200, type=int)
    parser.add_argument('--clip-max-norm', default=0.1, type=float, help='gradient clipping max norm')
    
    # Model Parameters
    parser.add_argument(
      '--frozen-weights',
      type=str,
      default=None,
      help='Path to the pretrained model. If set, only the mask head will be trained'
    )
    parser.add_argument('--num-queries', default=100, type=int, help='Number of query slots')
    
    # Backbone
    parser.add_argument('--backbone', default='resnet50', type=str, help='Name of CNN backbone to use')
    parser.add_argument(
      '--dilation', action='store_true', help='If true, we replace stride with dilation in the last conv block'
    )
    parser.add_argument(
      '--position-embedding',
      default='learned',
      type=str,
      choices=('learned', 'absolute', 'relative'),
      help='Type of positional encoding method to use on top of the image features'
    )
    
    # Transformer
    parser.add_argument('--enc-layers', default=6, type=int, help='Number of encoding layers in the transformer')
    parser.add_argument('--dec-layers', default=6, type=int, help='Number of decoding layers in the transformer')
    parser.add_argument(
      '--num-hidden-features', default=2048, type=int, help='Intermediate size of FFN in the transformer'
    )
    parser.add_argument('--embed-dim', default=256, type=int, help='Size of embedding dimension')
    parser.add_argument('--dropout', default=0.0, type=float, help='Dropout applied in the transformer')
    parser.add_argument(
      '--num-heads', default=8, type=int, help='Number of attention heads inside the transformer\'s attentions'
    )
    parser.add_argument('--pre-norm', action='store_true')
    parser.add_argument('--qkv-bias', action='store_true')
    
    # Segmentation
    parser.add_argument('--masks', action='store_true', help='Train segmentation head if this flag is provided')
    
    # Loss
    parser.add_argument(
      '--no-aux-loss',
      dest='aux_loss',
      action='store_false',
      help='Disable auxiliary decoding losses (loss at each layer)'
    )
    
    # Loss coefficients
    parser.add_argument('--bbox-loss-coef', default=5, type=float)
    parser.add_argument('--giou-loss-coef', default=2, type=float)
    parser.add_argument('--dice-loss-coef', default=1, type=float)
    parser.add_argument('--mask-loss-coef', default=1, type=float)
    parser.add_argument(
      '--eos-coef', default=0.1, type=float, help='Relative classification weight of the `no-object` class'
    )
    
    # Matcher
    parser.add_argument('--set-cost-class', default=1, type=float, help='Class coefficient in the matching cost')
    parser.add_argument('--set-cost-bbox', default=5, type=float, help='L1 bbox coefficient in the matching cost')
    parser.add_argument('--set-cost-giou', default=2, type=float, help='GIoU bbox coefficient in the matching cost')
    
    # Dataset Parameters
    parser.add_argument('--dataset-file', default='coco')
    
    # Misc
    parser.add_argument('--device', default='cuda', help='Device to use for training / inference')
    
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int, help='Number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='URL used to setup distributed training')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.num_classes = 91
    m, criterion, processor = build_model(args)
    m = m.cuda()
    dummy = torch.randn(1, 3, 1024, 1024).cuda()
    out = m(dummy)
    print(out['pred_logits'].shape)
