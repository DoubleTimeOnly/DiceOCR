from unittest import TestCase
from detector.lowrankprojector import LowRankProjector
import numpy as np
import cv2
from utils.boundingboxes import draw_boxes, show_boxes

class TestLowRankProjector(TestCase):
    def test_run_inference_random(self):
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        model = LowRankProjector()
        output = model(image)
        assert output["boxes"].size(dim=1) == 4

    def test_run_d20(self):
        image = cv2.imread("../datasets/test/d20_color001.jpg")
        model = LowRankProjector()
        output = model(image, copy_image=True)
        show_boxes(image, output["boxes"][:2])
        assert output["boxes"].size(dim=1) == 4

