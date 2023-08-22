import os
from typing import Optional, List, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision


def _max_by_axis(the_list: List[List[int]]) -> List[int]:
    """Find max value across dimension
    :param the_list: List[List[int]]
    :return: List[int]
    """
    maxes = the_list[0]
    for sublist in the_list[1:]:  # iterate through elements
        for index, item in enumerate(sublist):  # iterate through each dimension of the current element
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    """Create NestedTensor with tensor and it's mask"""
    def __init__(self, tensor: torch.Tensor, mask: Optional[torch.Tensor]) -> None:
        self.tensor = tensor
        self.mask = mask
    
    def to(self, device):
        cast_tensor = self.tensor.to(device)
        cast_mask = self.mask.to(device) if self.mask is not None else None
        return NestedTensor(cast_tensor, cast_mask)
    
    def decompose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.tensor, self.mask
    
    def __repr__(self) -> str:
        return str(self.tensor)


def nested_tensor_from_tensor_list(tensor_list: List[torch.Tensor]) -> NestedTensor:
    """Create a NestedTensor instance from a list of tensors

    :param tensor_list: List of Tensor
    :return: NestedTensor instance
    """
    if tensor_list[0].ndim != 3:
        raise ValueError('Image\'s shape not supported')
    if torchvision._is_tracing():
        # Using ONNX Handler
        return _onnx_nested_tensor_from_tensor_list(tensor_list)
    # For supporting different-sized images
    max_size = _max_by_axis([list(img.shape) for img in tensor_list])
    B, (C, H, W) = len(tensor_list), max_size
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    tensor = torch.zeros((B, C, H, W), dtype=dtype, device=device)  # empty tensor to pad
    mask = torch.ones((B, H, W), dtype=torch.bool, device=device)  # empty mask filled with True
    for img, pad_img, m in zip(tensor_list, tensor, mask):  # pad the image and its mask
        pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
        m[:img.shape[1], :img.shape[2]] = False
    return NestedTensor(tensor, mask)


# Only used when ONNX is tracing
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[torch.Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)
    
    padded_images, padded_masks = [], []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = F.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_images.append(padded_img)
        
        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = F.pad(m, (0, padding[2], 0, padding[1]), 'constant', 1)
        padded_masks.append(padded_mask)
    
    tensor, mask = torch.stack(padded_images), torch.stack(padded_masks)
    
    return NestedTensor(tensor, mask)


def setup_for_distributed(is_master) -> None:
    """This function disables printing when not in the master process
    
    :param is_master: boolean - master flag
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print
    
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    
    __builtin__.print = print


def is_dist_available_and_initialized() -> bool:
    return False if dist.is_available() else not dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_available_and_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if is_dist_available_and_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def save_on_master(*args, **kwargs) -> None:
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args) -> None:
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return
    
    args.distributed = True
    
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    dist.init_process_group(
      backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int] = (1, )) -> List[torch.Tensor]:
    """Computes the precision@k for the specified values of k

    :param output: torch.Tensor - output tensor
    :param target: torch.Tensor - target tensor
    :param topk: Tuple[int] - top k index
    :return: a list of precision
    """
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
