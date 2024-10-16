import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class KITTIRoadDataset(Dataset):
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
        mask = np.all(mask == [255, 0, 255], axis=-1).astype(np.uint8)

        if self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask)

        mask = torch.unsqueeze(mask, 0).float()
        return image, mask

def get_kitti_road_dataloader(config, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])
    
    dataset = KITTIRoadDataset(
        images_path=config.KITTI_ROAD_IMAGES_PATH,
        masks_path=config.KITTI_ROAD_MASKS_PATH,
        transform=transform,
        mask_transform=transform
    )
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=config.PIN_MEMORY)
    return loader
