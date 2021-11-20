import cv2
from detector.lowrankprojector import LowRankProjector
from utils import boundingboxes

def main():
    model = LowRankProjector()
    webcam = cv2.VideoCapture(0)

    while True:
        _, image = webcam.read()
        output = model(image, copy_image=True)
        boundingboxes.show_boxes(image, output["boxes"], duration=1)


if __name__ == "__main__":
    main()