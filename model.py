## code using another model to fix warining overdate pretrained=True
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_Weights
)


def build_model(backbone: str, num_classes: int):
    if backbone == 'fasterrcnn_resnet50_fpn':
        # Dùng weights chuẩn DEFAULT, không cần pretrained=True nữa
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    else:
        # MobileNet v3
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    
    # Lấy số feature từ ROI heads và thay box predictor
    in_feature = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feature, num_classes=num_classes)

    return model


