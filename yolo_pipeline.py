import os
import random
import torch
import cv2
from ultralytics import YOLO
from torchvision.utils import draw_bounding_boxes
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from dataset_utils import BIRDSAIYOLODataset

TEST_IMG_DIR = '/kaggle/working/yolo_birdsai/images/test'
TEST_LABEL_DIR = '/kaggle/working/yolo_birdsai/labels/test'

def train_baseline():
    yaml_path = '/kaggle/working/yolo_birdsai/dataset.yaml'
    output_dir = '/kaggle/working/runs/train'
    
    model = YOLO('yolov8m.pt') 
    model.train(
        data=yaml_path, epochs=30, imgsz=640, batch=16,
        device=[0,1], project=output_dir, name='baseline_birdsai',
        plots=True, save=True
    )

def test_evaluation_yolo(weights_path, conf_threshold=0.25):
    model = YOLO(weights_path)
    test_dataset = BIRDSAIYOLODataset(TEST_IMG_DIR, TEST_LABEL_DIR) 
    
    metric_animals = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5])
    metric_humans = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5])
    
    for idx in tqdm(range(len(test_dataset)), desc="Evaluating YOLO Test Set"):
        _, target = test_dataset[idx] 
        img_name = test_dataset.img_names[idx]
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        
        results = model(img_path, verbose=False)[0]
        p_boxes = results.boxes.xyxy.cpu() 
        p_scores = results.boxes.conf.cpu()
        p_labels = results.boxes.cls.cpu().int() + 1
        
        t_boxes = target['boxes'].cpu()
        t_labels = target['labels'].cpu()
        
        a_pred_mask, a_targ_mask = p_labels == 1, t_labels == 1
        metric_animals.update([{"boxes": p_boxes[a_pred_mask], "scores": p_scores[a_pred_mask], "labels": p_labels[a_pred_mask]}], [{"boxes": t_boxes[a_targ_mask], "labels": t_labels[a_targ_mask]}])
        
        h_pred_mask, h_targ_mask = p_labels == 2, t_labels == 2
        metric_humans.update([{"boxes": p_boxes[h_pred_mask], "scores": p_scores[h_pred_mask], "labels": p_labels[h_pred_mask]}], [{"boxes": t_boxes[h_targ_mask], "labels": t_labels[h_targ_mask]}])

    print(metric_animals.compute())
    print(metric_humans.compute())

if __name__ == '__main__':
    # train_baseline()
    test_evaluation_yolo('/kaggle/input/models/zayeemb/yolo/pytorch/default/1/best.pt')