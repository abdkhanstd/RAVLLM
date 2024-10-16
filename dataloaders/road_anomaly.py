import os
import cv2
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class RoadAnomalyDataset(Dataset):
    def __init__(self, images_path, annotations_path, transform=None):
        self.images_path = images_path
        self.annotations_path = annotations_path
        self.image_files = [
            os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) 
            for f in filenames if f.endswith('.jpg') and os.path.exists(os.path.join(annotations_path, f.replace('.jpg', '.labels/index_.json')))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_name = os.path.basename(img_path)
        img_name = img_name.replace('.jpg', '')
        annotation_path = os.path.join(self.annotations_path, img_name + ".labels", "index_.json")

        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        original_size = image.shape[:2]  # (height, width)

        if self.transform:
            image = self.transform(image)

        bboxes = []
        for obj in annotation['instances']:
            roi_rect = obj['roi_rect']
            bboxes.append([
                [roi_rect[0][0], roi_rect[0][1]],  # Top-left corner
                [roi_rect[1][0], roi_rect[1][1]]   # Bottom-right corner
            ])

        return image, bboxes, original_size

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    bboxes = [item[1] for item in batch]
    original_sizes = [item[2] for item in batch]
    return images, bboxes, original_sizes

def get_road_anomaly_dataloader(config, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])
    
    dataset = RoadAnomalyDataset(
        images_path=config.ROAD_ANOMALY_IMAGES_PATH,
        annotations_path=config.ROAD_ANOMALY_ANNOTATIONS_PATH,
        transform=transform
    )
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=config.PIN_MEMORY, collate_fn=collate_fn)
    return loader

def show_samples(dataloader, num_samples=10):
    for i, (images, bboxes, original_sizes) in enumerate(dataloader):
        if i >= num_samples:
            break
        for j in range(len(images)):
            img = images[j].permute(1, 2, 0).numpy()
            original_size = original_sizes[j]

            # Resize the image back to its original size
            img_resized = cv2.resize(img, (original_size[1], original_size[0]))

            bbox = bboxes[j]
            plt.figure()
            plt.imshow(img_resized)
            for box in bbox:
                rect = plt.Rectangle((box[0][0], box[0][1]), box[1][0] - box[0][0], box[1][1] - box[0][1], fill=False, color='red')
                plt.gca().add_patch(rect)
            plt.show()

if __name__ == "__main__":
    class Config:
        ROAD_ANOMALY_IMAGES_PATH = '../datasets/RA/images'
        ROAD_ANOMALY_ANNOTATIONS_PATH = '../datasets/RA/masks'
        PIN_MEMORY = True

    config = Config()
    road_anomaly_loader = get_road_anomaly_dataloader(config, batch_size=1)
    show_samples(road_anomaly_loader, num_samples=10)
