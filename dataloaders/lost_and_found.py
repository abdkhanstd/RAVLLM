import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np

class LostAndFoundDataset(Dataset):
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
        road_mask = np.all(mask == [128, 64, 128], axis=-1).astype(np.uint8)

        if self.transform:
            image = self.transform(image)
            road_mask = self.mask_transform(road_mask)

        road_mask = torch.unsqueeze(road_mask, 0).float()
        return image, road_mask

def get_lost_and_found_dataloader(config, batch_size, mode='Train'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])
    
    if mode == 'Train':
        dataset = LostAndFoundDataset(
            images_path=config.LOST_AND_FOUND_TRAIN_IMAGES_PATH,
            masks_path=config.LOST_AND_FOUND_TRAIN_MASKS_PATH,
            transform=transform,
            mask_transform=transform
        )
    else:
        dataset = LostAndFoundDataset(
            images_path=config.LOST_AND_FOUND_TEST_IMAGES_PATH,
            masks_path=config.LOST_AND_FOUND_TEST_MASKS_PATH,
            transform=transform,
            mask_transform=transform
        )
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=config.PIN_MEMORY)
    return loader
