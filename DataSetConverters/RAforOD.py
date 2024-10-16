import os
import cv2
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from PIL import Image
from torch.utils.data import Dataset

def generate_index_json(base_path, output_path):
    images_path = os.path.join(base_path, 'images')
    masks_path = os.path.join(base_path, 'masks')
    
    os.makedirs(output_path, exist_ok=True)
    jsonl_data = []

    for mode in ['test']:
        img_dir = os.path.join(images_path)
        mask_dir = os.path.join(masks_path)
        
        for subfolder in os.listdir(img_dir):
            img_subdir = os.path.join(img_dir, subfolder)
            mask_subdir = os.path.join(mask_dir, subfolder)
            if not os.path.isdir(img_subdir):
                continue

            for file in os.listdir(img_subdir):
                if file.endswith('.png'):
                    img_name = file
                    mask_name = img_name.replace('leftImg8bit_', '').replace('.png', '_gtCoarse_color.png')
                    mask_path = os.path.join(mask_subdir, mask_name).replace('_leftImg8bit','')
                    
                    if not os.path.exists(mask_path):
                        continue

                    mask = cv2.imread(mask_path)
                    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                    target_color = [0, 0, 142]  # RGB color to detect

                    # Find all bounding boxes
                    mask_bin = np.all(mask_rgb == target_color, axis=-1).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    instances = []
                    suffix_parts = []
                    for i, contour in enumerate(contours):
                        x, y, w, h = cv2.boundingRect(contour)
                        if w > 0 and h > 0:  # Filter out zero area boxes
                            roi_rect = [[x, y], [x+w, y+h]]
                            instances.append({
                                "id": i+1,
                                "cls": 2,
                                "crop_rect": roi_rect,
                                "roi_rect": roi_rect
                            })
                            suffix_parts.append(f"<loc_{x}><loc_{y}><loc_{x+w}><loc_{y+h}>")

                    # Debugging information
                    if len(instances) == 0:
                        print(f"No instances found in mask: {mask_path}")
                    else:
                        print(f"Found {len(instances)} instances in mask: {mask_path}")
                    
                    annotation = {"instances": instances}
                    
                    json_path = os.path.join(output_path, img_name.replace('leftImg8bit_', '').replace('.png', '_index.json'))
                    with open(json_path, 'w') as json_file:
                        json.dump(annotation, json_file, indent=4)
                    print(f"Generated {json_path}")

                    jsonl_data.append({
                        "image": os.path.relpath(os.path.join(img_subdir, img_name), base_path),
                        "prefix": "<OD>",
                        "suffix": "".join(suffix_parts)
                    })

    jsonl_output_path = os.path.join(output_path, 'annotations.jsonl')
    with open(jsonl_output_path, 'w') as jsonl_file:
        for item in jsonl_data:
            jsonl_file.write(json.dumps(item) + '\n')
    print(f"Generated {jsonl_output_path}")

# Base path
base_path = 'RoadAnomaly'
output_path = 'RoadAnomaly_new'
generate_index_json(base_path, output_path)


class JSONLDataset:
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        self.jsonl_file_path = jsonl_file_path
        self.image_directory_path = image_directory_path
        self.entries = self._load_entries()

    def _load_entries(self) -> List[Dict[str, Any]]:
        entries = []
        with open(self.jsonl_file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                entries.append(data)
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")

        entry = self.entries[idx]
        image_path = os.path.join(self.image_directory_path, entry['image'])
        try:
            image = Image.open(image_path)
            return (image, entry)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file {image_path} not found.")


class DetectionDataset(Dataset):
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        self.dataset = JSONLDataset(jsonl_file_path, image_directory_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, data = self.dataset[idx]
        prefix = data['prefix']
        suffix = data['suffix']
        return prefix, suffix, image
