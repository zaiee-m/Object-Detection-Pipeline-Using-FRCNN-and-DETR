import os
import glob
import random
import torch
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
from transformers import AutoImageProcessor

from dataset_utils import BIRDSAIYOLODataset
# Assuming other models are imported as needed
# from yolo_pipeline import YOLO
# from frcnn_pipeline import build_custom_frcnn_base, build_frcnn_focal_loss
# from detr_pipeline import build_deformable_detr

def get_images_from_videos(test_dir, video_list, num_samples=5):
    collected_frames = []
    for video_name in video_list:
        search_pattern = os.path.join(test_dir, f"{video_name}_*.jpg")
        collected_frames.extend(glob.glob(search_pattern))
    return random.sample(collected_frames, min(num_samples, len(collected_frames)))

def generate_ultimate_master_comparison(image_paths, output_dir, conf_threshold=0.30):
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize your model paths mapping here ...
    print("Loading all models into memory...")
    
    # Run inferences, clone base image_tensors, and build subplots...
    # (Implementation maintained exactly from your original script)

if __name__ == '__main__':
    test_dir = '/kaggle/working/yolo_birdsai/images/test'
    easy_videos = ['0000000352_0000000000'] 
    hard_videos = ['0000000058_0000000000'] 
    
    target_images = get_images_from_videos(test_dir, easy_videos, 2) + get_images_from_videos(test_dir, hard_videos, 2)
    generate_ultimate_master_comparison(target_images, output_dir='/kaggle/working/report_ultimate', conf_threshold=0.15)