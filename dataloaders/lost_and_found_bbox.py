import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

class LostAndFoundAnomalyDataset(Dataset):
    def __init__(self, images_path, annotations_path, transform=None):
        self.images_path = images_path
        self.annotations_path = annotations_path
        
        self.image_files = [
            os.path.join(dp, f) 
            for dp, dn, filenames in os.walk(images_path) 
            for f in filenames 
            if f.endswith('.png') and os.path.exists(os.path.join(annotations_path, os.path.basename(dp), f.replace('leftImg8bit_', '').replace('.png', '_index.json')))
        ]

        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_name = os.path.basename(img_path)
        subfolder = os.path.basename(os.path.dirname(img_path))
        annotation_path = os.path.join(self.annotations_path, subfolder, img_name.replace('leftImg8bit_', '').replace('.png', '_index.json'))

        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        image = Image.open(img_path).convert("RGB")
        original_width, original_height = image.size

        if self.transform:
            image = self.transform(image)
        
        if isinstance(image, torch.Tensor):
            image_size = (image.shape[2], image.shape[1])  # (width, height)
        else:
            image_size = image.size

        # Convert to the Florence2 expected format
        bboxes = ""
        for obj in annotation['instances']:
            label = obj['cls']
            roi_rect = obj['roi_rect']
            x_scale = image_size[0] / original_width
            y_scale = image_size[1] / original_height
            x_min, y_min = int(roi_rect[0][0] * x_scale), int(roi_rect[0][1] * y_scale)
            x_max, y_max = int(roi_rect[1][0] * x_scale), int(roi_rect[1][1] * y_scale)
            bboxes += f"{label}<loc_{x_min}><loc_{y_min}><loc_{x_max}><loc_{y_max}>"

        return {'image': image, 'bboxes': bboxes, 'image_size': image_size}

def collate_fn(batch):
    images = [item['image'] for item in batch]
    bboxes = [item['bboxes'] for item in batch]
    return images, bboxes

def get_lost_and_found_anomaly_dataloader(config, batch_size, mode='train'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    if mode == 'train':
        images_path = config.LostAndFound_TRAIN_IMAGES_PATH
        annotations_path = config.LostAndFound_TRAIN_MASKS_PATH
    else:
        images_path = config.LostAndFound_TEST_IMAGES_PATH
        annotations_path = config.LostAndFound_TEST_MASKS_PATH

    dataset = LostAndFoundAnomalyDataset(
        images_path=images_path,
        annotations_path=annotations_path,
        transform=transform
    )
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode=='train'), collate_fn=collate_fn)
    return loader