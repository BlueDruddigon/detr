from argparse import Namespace

from .detr import build


def build_model(args: Namespace):
    return build(args)
