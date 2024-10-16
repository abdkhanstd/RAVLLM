import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class Vkitti(Dataset):
    def __init__(self, images_path, masks_path, transform=None, mask_transform=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.image_files = [f for f in os.listdir(images_path) if f.endswith('.jpg')]
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = img_name.replace('.jpg', '.png')
        img_path = os.path.join(self.images_path, img_name)
        mask_path = os.path.join(self.masks_path, mask_name)

        # Load image and convert to RGB
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask and convert the relevant regions to binary mask (0 and 255)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8) * 255  # Convert to binary mask with 0 and 255

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = torch.unsqueeze(torch.tensor(mask, dtype=torch.float32), 0)
        return image, mask

class RoadDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None, mask_transform=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.image_files = os.listdir(images_path)
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = img_name.replace('_', '_road_')
        img_path = os.path.join(self.images_path, img_name)
        mask_path = os.path.join(self.masks_path, mask_name)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path)
        mask = np.all(mask == [255, 0, 255], axis=-1).astype(np.uint8) * 255  # Binary mask with 0 and 255

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = torch.unsqueeze(torch.tensor(mask, dtype=torch.float32), 0)
        return image, mask

class RA2RoadDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None, mask_transform=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.image_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames if f.endswith('.png')]
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_name = os.path.basename(img_path)
        subfolder = os.path.basename(os.path.dirname(img_path))
        mask_path = os.path.join(self.masks_path, subfolder, img_name.replace('.png', '_gtCoarse_color.png'))
        mask_path = mask_path.replace('_leftImg8bit', '')

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path)
        road_mask = np.all(mask == [128, 64, 128], axis=-1).astype(np.uint8) * 255  # Binary mask with 0 and 255

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            road_mask = self.mask_transform(road_mask)

        road_mask = torch.unsqueeze(torch.tensor(road_mask, dtype=torch.float32), 0)
        return image, road_mask

class CombinedRoadDataset(Dataset):
    def __init__(self, vkitti_images_path, vkitti_masks_path, road_images_path, road_masks_path, ra2_images_path, ra2_masks_path, transform=None, mask_transform=None):
        self.vkitti_dataset = Vkitti(vkitti_images_path, vkitti_masks_path, transform, mask_transform)
        self.road_dataset = RoadDataset(road_images_path, road_masks_path, transform, mask_transform)
        self.ra2_road_dataset = RA2RoadDataset(ra2_images_path, ra2_masks_path, transform, mask_transform)
        self.datasets = [self.vkitti_dataset, self.road_dataset, self.ra2_road_dataset]
        self.lengths = [len(self.vkitti_dataset), len(self.road_dataset), len(self.ra2_road_dataset)]

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        if idx < self.lengths[0]:
            return self.vkitti_dataset[idx]
        elif idx < sum(self.lengths[:2]):
            return self.road_dataset[idx - self.lengths[0]]
        else:
            return self.ra2_road_dataset[idx - sum(self.lengths[:2])]

def get_dataloaders(config, road_batch_size, anomaly_batch_size, mode='Train'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST)
    ])

    print("Using combined Datasets for Road Training")
    road_dataset = CombinedRoadDataset(
        vkitti_images_path=config.VKITTI_IMAGES_PATH,
        vkitti_masks_path=config.VKITTI_MASKS_PATH,
        road_images_path=config.KITTIRoad_TRAINING_IMAGES_PATH,
        road_masks_path=config.KITTIRoad_TRAINING_MASKS_PATH,
        ra2_images_path=config.LostAndFound_TRAIN_IMAGES_PATH,
        ra2_masks_path=config.LostAndFound_TRAIN_MASKS_PATH,
        transform=transform,
        mask_transform=mask_transform
    )
    if len(road_dataset) == 0:
        raise ValueError("The road dataset is empty.")

    road_loader = DataLoader(road_dataset, batch_size=road_batch_size, shuffle=True, pin_memory=config.PIN_MEMORY)
    return road_loader
