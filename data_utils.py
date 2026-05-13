import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image

class BIRDSAIYOLODataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_names = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.img_names.sort()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        img_tensor = TF.to_tensor(img)

        boxes, labels = [], []

        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5: continue
                    
                    yolo_class = int(parts[0])
                    x_center, y_center, w_norm, h_norm = map(float, parts[1:])
                    
                    x_min = max(0.0, min((x_center - (w_norm / 2)) * img_w, float(img_w)))
                    y_min = max(0.0, min((y_center - (h_norm / 2)) * img_h, float(img_h)))
                    x_max = max(0.0, min((x_center + (w_norm / 2)) * img_w, float(img_w)))
                    y_max = max(0.0, min((y_center + (h_norm / 2)) * img_h, float(img_h)))
                    
                    if x_max > x_min and y_max > y_min:
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(yolo_class + 1)

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.empty((0, 4), dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64) if len(boxes) > 0 else torch.empty((0,), dtype=torch.int64)

        target = {"boxes": boxes_tensor, "labels": labels_tensor, "image_id": torch.tensor([idx])}
        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))