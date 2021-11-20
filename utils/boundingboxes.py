from torchvision.utils import draw_bounding_boxes
from utils.images import to_tensor_and_CHW, show_tensor
import numpy as np
import torch


def draw_boxes(image, boxes):
    if isinstance(image, np.ndarray):
        image = torch.tensor(image)
        image = image.permute(2, 0, 1)
    colors = ["blue", "yellow"]
    canvas = draw_bounding_boxes(image, boxes, width=3)
    return canvas


def show_boxes(image, boxes, name="tensor", duration=0):
    canvas_tensor = draw_boxes(image, boxes)
    show_tensor(canvas_tensor, name=name, duration=duration)
