import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from torchvision.models.detection import FasterRCNN, roi_heads
from torchvision.models.detection.rpn import AnchorGenerator
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from dataset_utils import BIRDSAIYOLODataset, collate_fn

class CustomResNet50FPN(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        self.body = IntermediateLayerGetter(resnet, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(in_channels_list=[256, 512, 1024, 2048], out_channels=256)
        self.out_channels = 256

    def forward(self, x):
        return self.fpn(self.body(x))

def build_custom_frcnn_base(num_classes=3):
    backbone = CustomResNet50FPN()
    anchor_sizes = ((32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)
    
    return FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_anchor_generator=rpn_anchor_generator, box_roi_pool=roi_pooler)

# --- Focal Loss Override ---
original_fastrcnn_loss = roi_heads.fastrcnn_loss

def focal_loss(class_logits, labels, alpha=0.25, gamma=2.0):
    ce_loss = F.cross_entropy(class_logits, labels, reduction='none')
    pt = torch.exp(-ce_loss)
    return (alpha * (1 - pt)**gamma * ce_loss).mean()

def custom_focal_fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    labels_cat = torch.cat(labels, dim=0)
    classification_loss = focal_loss(class_logits, labels_cat)
    _, original_box_loss = original_fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
    return classification_loss, original_box_loss

def build_frcnn_focal_loss(num_classes=3):
    model = build_custom_frcnn_base(num_classes=num_classes)
    roi_heads.fastrcnn_loss = custom_focal_fastrcnn_loss
    return model

# Add your standard test_evaluation_frcnn & train_one_epoch implementations here.