import cv2
import torch
import numpy as np
import torchvision.transforms as transform
import torchvision


def preprocess_image(image):
    '''
    Convert a numpy array into torch tensor
    Transposes channels
    :param image:
    :return:
    '''
    if isinstance(image, np.ndarray):
        image = torch.tensor(image)
    image = torch.transpose(image, -1, 0)   # Torch uses CxHxW
    pipeline = torchvision.transforms.Compose([
        transform.ConvertImageDtype(torch.float),
        transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return pipeline(image)


def show_tensor(tensor, name="tensor", duration=0):
    canvas = torch.transpose(tensor, 0, -1).numpy()
    cv2.imshow(name, canvas)
    cv2.waitKey(duration)

