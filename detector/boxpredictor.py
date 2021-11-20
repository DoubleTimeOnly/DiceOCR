import torch
from torchvision.models.detection import image_list
from torchvision.models.detection.faster_rcnn import TwoMLPHead
from torchvision.models.detection.backbone_utils import mobilenet_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead, concat_box_prediction_layers
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops.poolers import MultiScaleRoIAlign
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_mobilenet_v3_large_320_fpn

class BoxPredictor():
    def __init__(self, useCuda=False):
        super().__init__()
        self.useCuda = useCuda
        if self.useCuda:
            self.device = torch.device("cuda")
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
        self.backbone = mobilenet_backbone("mobilenet_v3_large", pretrained=True, fpn=True, trainable_layers=0)
        self.backbone.load_state_dict(self.model.backbone.state_dict())
        anchor_sizes = ((32, 64, 128, 256, 512,),) * 3
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.3
        rpn_batch_size_per_image = 256
        rpn_positive_fraction = 0.5
        rpn_pre_nms_top_n_train = 2000
        rpn_post_nms_top_n_train = 2000
        rpn_pre_nms_top_n_test = 10
        rpn_post_nms_top_n_test = 10
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        rpn_nms_thresh = 0.7
        rpn_score_thresh = 0.05
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        rpn_head = RPNHead(self.backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0] )
        self.rpn = RPN(rpn_anchor_generator, rpn_head,
                       rpn_fg_iou_thresh, rpn_bg_iou_thresh,
                       rpn_batch_size_per_image, rpn_positive_fraction,
                       rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
                       score_thresh=rpn_score_thresh )
        weighted_state_dict = self.model.rpn.state_dict()
        self.rpn.load_state_dict(self.model.rpn.state_dict())

        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        # min_size, max_size = (800, 1333)
        min_size, max_size = (320, 640)
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        self.box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2)
        if self.useCuda:
            self.backbone.to(self.device)
            self.rpn.to(self.device)
            self.box_roi_pool.to(self.device)
        self.backbone.eval()
        self.rpn.eval()
        self.box_roi_pool.eval()

    def __call__(self, images):
        '''

        :param image_batch: tensor of shape NCHW
        :return:
        '''
        targets = None
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        if self.useCuda:
            images = [image.to(self.device) for image in images]
        images, targets = self.transform(images, targets)


        resolution = self.box_roi_pool.output_size[0]
        representation_size = 1024
        # box_head = TwoMLPHead(
        #     self.backbone.out_channels * resolution ** 2,
        #     representation_size)

        backbone_features = self.backbone(images.tensors)
        proposals, scores = self.rpn(images, backbone_features, targets)
        box_features = None
        box_features = self.box_roi_pool(backbone_features, proposals, images.image_sizes)
        # if self.useCuda:
        #     box_features = box_features.to(self.device)
        #     box_head.to(self.device)
        # box_features = box_head(box_features)
        if self.useCuda:
            proposals = [p.cpu() for p in proposals]
            scores = [s.cpu() for s in scores]
            # box_features = [feat.cpu() for feat in box_features]
            # box_features = box_features.cpu()
        return proposals, scores, box_features, images
        # detections, detector_losses = self.roi_heads(backbone_features, proposals, images.image_sizes, targets)
        # detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        # return detections

class RPN(RegionProposalNetwork):
    def forward(self,
                images,  # type: ImageList
                features,  # type: Dict[str, Tensor]
                targets=None  # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Tensor], Dict[str, Tensor]]
        """
        Args:
            images (ImageList): images for which we want to compute the predictions
            features (OrderedDict[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[Tensor]]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = \
            concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        return boxes, scores
