from torchvision.utils import draw_bounding_boxes
from utils.images import preprocess_image, show_tensor
import numpy as np
import torch


def draw_boxes(image, boxes):
    if isinstance(image, np.ndarray):
        image = torch.tensor(image)
        image = torch.transpose(image, -1, 0)
    colors = ["blue", "yellow"]
    canvas = draw_bounding_boxes(image, boxes, width=3)
    return canvas


def show_boxes(image, boxes, name="tensor", duration=0):
    canvas_tensor = draw_boxes(image, boxes)
    show_tensor(canvas_tensor, name=name, duration=duration)
