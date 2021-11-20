import numpy as np
import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection import roi_heads, FasterRCNN
from utils.images import to_tensor_and_CHW


class LowRankProjector:
    def __init__(self, useCuda=False):
        self.model = ssdlite320_mobilenet_v3_large(pretrained=True)
        # self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        # self.model = retinanet_resnet50_fpn(pretrained=True)
        # self.model = FasterRCNN()
        self.model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
        self.model.eval()
        self.useCuda = useCuda
        self.device = None
        if self.useCuda:
            self.device = torch.device("cuda")
            self.model.to(self.device)

    def __call__(self, image, copy_image=True):
        inference_image = image.copy()
        inference_image = torch.tensor(inference_image)
        inference_image = to_tensor_and_CHW(inference_image)
        inference_image = torch.unsqueeze(inference_image, 0)
        if self.useCuda:
            inference_image = inference_image.cuda()
        output = self.model(inference_image)[0]
        if self.useCuda:
            for key in output:
                output[key] = output[key].cpu()
        output = self.threshold(output, 0.05)
        output = self.size_filter(output, max_width=0.7 * inference_image.shape[3], max_height=0.7 * inference_image.shape[2])
        output = self.nms(output, iou_threshold=0.4)
        return output

    def size_filter(self, model_output, max_width, max_height):
        good_indices = []
        for idx, box in enumerate(model_output["boxes"]):
            width = box[2] - box[0]
            height = box[3] - box[1]
            if not (width > max_width or height > max_height):
                good_indices.append(idx)
        good_indices = torch.tensor(good_indices)
        model_output = self.sort_model_output(model_output, good_indices)
        return model_output


    def threshold(self, model_output, threshold):
        good_indices = (model_output["scores"] >= threshold).nonzero()
        good_indices = torch.squeeze(good_indices, 1)
        model_output = self.sort_model_output(model_output, good_indices)
        return model_output

    def nms(self, model_output, iou_threshold):
        good_indices = torchvision.ops.nms(model_output["boxes"], model_output["scores"], iou_threshold=iou_threshold)
        model_output = self.sort_model_output(model_output, good_indices)
        return model_output

    def resize_normalize_image(self, image_tensor):
        pipeline = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(320, 320))
        ])
        image_tensor = pipeline(image_tensor)
        return image_tensor

    def sort_model_output(self, model_output, indices):
        if len(indices) == 0:
            model_output["boxes"] = torch.zeros(size=[0, 4], dtype=torch.float)
            model_output["labels"] = torch.zeros(size=[0], dtype=torch.float)
            model_output["scores"] = torch.zeros(size=[0], dtype=torch.float)
            return model_output

        for output_type in ["boxes", "labels", "scores"]:
            model_output[output_type] = torch.index_select(model_output[output_type], 0, indices)
        return model_output
