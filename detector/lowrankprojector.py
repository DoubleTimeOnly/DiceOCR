import numpy as np
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from utils.images import preprocess_image


class LowRankProjector:
    def __init__(self):
        self.model = ssdlite320_mobilenet_v3_large(pretrained=True)

    def __call__(self, image, copy_image=True):
        image = torch.tensor(image)
        image = preprocess_image(image)
        image = torch.unsqueeze(image, 0)
        self.model.eval()
        output = self.model(image)
        return output[0]