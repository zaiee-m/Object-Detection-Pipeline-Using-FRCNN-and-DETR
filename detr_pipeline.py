import torch
import torch.nn.functional as F
from transformers import DeformableDetrForObjectDetection, AutoImageProcessor
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_utils import BIRDSAIYOLODataset, collate_fn

def build_deformable_detr(experiment="Pretrained", num_classes=3):
    model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr", num_labels=num_classes, ignore_mismatched_sizes=True)
    
    if experiment == "Pretrained":
        model.eval()
        return model

    for param in model.model.backbone.parameters(): param.requires_grad = False

    if experiment == "Exp2": # Decoder Only
        for param in model.model.encoder.parameters(): param.requires_grad = False
    elif experiment == "Exp3": # Encoder Only
        for param in model.model.decoder.parameters(): param.requires_grad = False
            
    for param in model.class_embed.parameters(): param.requires_grad = True
    for param in model.bbox_embed.parameters(): param.requires_grad = True
        
    return model

def format_detr_labels(targets, batched_images):
    detr_labels = []
    h, w = batched_images.shape[-2], batched_images.shape[-1]
    
    for t in targets:
        boxes = t["boxes"]
        if len(boxes) > 0:
            box_width = boxes[:, 2] - boxes[:, 0]
            box_height = boxes[:, 3] - boxes[:, 1]
            norm_cx = (boxes[:, 0] + (box_width / 2)) / w
            norm_cy = (boxes[:, 1] + (box_height / 2)) / h
            norm_w = box_width / w
            norm_h = box_height / h
            norm_boxes = torch.clamp(torch.stack([norm_cx, norm_cy, norm_w, norm_h], dim=-1), 0.0, 1.0)
        else:
            norm_boxes = torch.empty((0, 4), dtype=torch.float32)
            
        detr_labels.append({"class_labels": t["labels"], "boxes": norm_boxes})
    return detr_labels

def run_all_detr_experiments(train_img_dir, train_label_dir, num_epochs=10):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset = BIRDSAIYOLODataset(train_img_dir, train_label_dir)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn)
    
    experiments = ["Exp1", "Exp2", "Exp3"]
    for exp in experiments:
        model = build_deformable_detr(experiment=exp, num_classes=3).to(device)
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)
        
        # training loop implementation goes here...