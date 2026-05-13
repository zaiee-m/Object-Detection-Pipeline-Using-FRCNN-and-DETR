import os
import shutil
import glob
from pathlib import Path
import pandas as pd
import cv2
from tqdm import tqdm

# Configurations
KAGGLE_INPUT_ROOT = '/kaggle/input/datasets/zayeemb/birdsai'
OUTPUT_DIR = '/kaggle/working/yolo_birdsai'

MOT_COLUMNS = [
    'frame_number', 'object_id', 'x', 'y', 'w', 'h', 
    'class_id', 'species', 'occlusion', 'noise'
]

SPLITS = {
    'train': os.path.join(KAGGLE_INPUT_ROOT, 'conservation_drones_train_real', 'TrainReal'),
    'val': os.path.join(KAGGLE_INPUT_ROOT, 'conservation_drones_train_real', 'TrainReal'),
    'test': os.path.join(KAGGLE_INPUT_ROOT, 'conservation_drones_test_real', 'TestReal')
}

def setup_directories(base_dir):
    dirs = [
        f'{base_dir}/images/train', f'{base_dir}/images/val', f'{base_dir}/images/test',
        f'{base_dir}/labels/train', f'{base_dir}/labels/val', f'{base_dir}/labels/test'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"Created YOLO directory structure at {base_dir}")

def process_split(yolo_split_name, source_dir):
    annotations_dir = os.path.join(source_dir, 'annotations')
    images_dir = os.path.join(source_dir, 'images')
    csv_files = glob.glob(os.path.join(annotations_dir, '*.csv'))
    
    csv_files.sort() 
    split_index = int(len(csv_files) * 0.8)
    
    if yolo_split_name == 'train':
        csv_files = csv_files[:split_index]
    elif yolo_split_name == 'val':
        csv_files = csv_files[split_index:]
    
    image_lookup = {}
    for root, _, files in os.walk(images_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                parent_folder = os.path.basename(root)
                image_lookup[f"{parent_folder}/{file}"] = os.path.join(root, file)
                image_lookup[file] = os.path.join(root, file)
    
    for csv_path in csv_files:
        video_name = Path(csv_path).stem
        df = pd.read_csv(csv_path, names=MOT_COLUMNS)
        df = df[df['class_id'].isin([0, 1])] 
        
        if df.empty:
            continue
            
        grouped = df.groupby('frame_number')
        
        for frame_idx, group in tqdm(grouped, desc=f"Processing {video_name}", leave=False):
            target_filename = f"{video_name}_{int(frame_idx):010d}.jpg"
            src_img_path = image_lookup.get(f"{video_name}/{target_filename}", image_lookup.get(target_filename))
            
            if not src_img_path:
                continue 
            
            img = cv2.imread(src_img_path)
            if img is None:
                continue
            
            img_height, img_width = img.shape[:2]
            
            out_prefix = f"{video_name}_{int(frame_idx):06d}"
            dst_img_path = os.path.join(OUTPUT_DIR, 'images', yolo_split_name, f"{out_prefix}.jpg") 
            dst_label_path = os.path.join(OUTPUT_DIR, 'labels', yolo_split_name, f"{out_prefix}.txt")
            
            shutil.copy(src_img_path, dst_img_path)
        
            with open(dst_label_path, 'w') as f:
                for _, row in group.iterrows():
                    class_id = int(row['class_id'])
                    x_center = max(0.0, min(1.0, (row['x'] + (row['w'] / 2)) / img_width))
                    y_center = max(0.0, min(1.0, (row['y'] + (row['h'] / 2)) / img_height))
                    w_norm = max(0.0, min(1.0, row['w'] / img_width))
                    h_norm = max(0.0, min(1.0, row['h'] / img_height))
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

def generate_yaml():
    yaml_content = """
path: /kaggle/working/yolo_birdsai
train: images/train
val: images/val
test: images/test

names:
  0: Animal
  1: Human
"""
    yaml_path = f'{OUTPUT_DIR}/dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content.strip())
    print(f"Success! dataset.yaml created at: {yaml_path}")

if __name__ == '__main__':
    setup_directories(OUTPUT_DIR)
    for yolo_split, source_path in SPLITS.items():
        if os.path.exists(source_path):
            process_split(yolo_split, source_path)
    generate_yaml()