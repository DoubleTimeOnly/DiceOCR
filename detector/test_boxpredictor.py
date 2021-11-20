import glob
from unittest import TestCase
import cv2
from detector.boxpredictor import BoxPredictor
import torch
from torchvision.models.detection.image_list import ImageList
from utils.images import to_tensor_and_CHW
from utils.boundingboxes import show_boxes

class TestBoxPredictor(TestCase):
    def test_BoxPredictor(self):
        image = cv2.imread("../datasets/test/d20_color001.jpg")
        inference_image = image.copy()
        inference_image = torch.tensor(inference_image)
        inference_image = to_tensor_and_CHW(inference_image)
        # inference_image = torch.unsqueeze(inference_image, 0)
        rpn = BoxPredictor()
        images = [inference_image]
        detections, scores, box_features, images = rpn(images)
        image_sizes = images.image_sizes
        image = cv2.resize(image, dsize=image_sizes[0])
        show_boxes(image, detections[0][:1], duration=0)

    def test_all_dice_types(self):
        rpn = BoxPredictor(useCuda=False)
        for file in glob.glob("../datasets/dice/**/*.jpg", recursive=True):
            image = cv2.imread(file)
            inference_image = image.copy()
            inference_image = torch.tensor(inference_image)
            inference_image = to_tensor_and_CHW(inference_image)
            # inference_image = torch.unsqueeze(inference_image, 0)
            images = [inference_image]
            detections, scores, box_features, images = rpn(images)
            image_sizes = images.image_sizes
            image = cv2.resize(image, dsize=image_sizes[0])
            show_boxes(image, detections[0][:1], duration=1)


