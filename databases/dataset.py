from PIL.Image import Image
import torch
import torchvision
import torchvision.transforms as transforms
import glob
import os
from utils import logger

log = logger.get_logger(__name__)
log.setLevel(logger.DEBUG)

class ImageDatabase:
    def __init__(self, path):
        path = os.path.abspath(path)
        self.images = ImageDatabase.load_images(path)


    @staticmethod
    def load_images(path):
        extensions = ["png", "jpg", "bmp"]
        all_filenames= []
        images = []
        for ext in extensions:
            all_filenames.extend(glob.glob(path + '/**/*.' + ext, recursive=True))
        for image_name in all_filenames:
            image = torchvision.io.read_image(image_name)
            image = ImageDatabase.preprocess_image(image)
            images.append(image)
        return images


    @staticmethod
    def preprocess_image(image):
        preprocess_steps = torchvision.transforms.Compose([
            transforms.ConvertImageDtype(torch.float),
        ])
        image = preprocess_steps(image)
        return image

