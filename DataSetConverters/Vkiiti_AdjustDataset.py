#Convert vkiiti road dataset in to images and masks folder from original dataste (placed in masks and images folder)

import os
import shutil
import re
import cv2
import numpy as np
from tqdm import tqdm

Image_folder = 'images'
Mask_folder = 'masks'
New_Image_folder = 'new/images'
New_Mask_folder = 'new/masks'

# Create new directories if they do not exist
os.makedirs(New_Image_folder, exist_ok=True)
os.makedirs(New_Mask_folder, exist_ok=True)

# Loop over the items in the base directory with progress tracking
for L1 in tqdm(os.listdir(Image_folder), desc='Level 1', leave=False):  # Level 1 directory
    L1_path = os.path.join(Image_folder, L1)
    if os.path.isdir(L1_path):
        for L2 in tqdm(os.listdir(L1_path), desc='Level 2', leave=False):  # Level 2 directory
            L2_path = os.path.join(L1_path, L2)
            if os.path.isdir(L2_path):
                for L3 in tqdm(os.listdir(L2_path), desc='Level 3', leave=False):  # Level 3 directory
                    L3_path = os.path.join(L2_path, L3)
                    if os.path.isdir(L3_path):
                        for L4 in tqdm(os.listdir(L3_path), desc='Level 4', leave=False):  # Level 4 directory
                            L4_path = os.path.join(L3_path, L4)
                            if os.path.isdir(L4_path):
                                for L5 in tqdm(os.listdir(L4_path), desc='Level 5', leave=False):  # Level 5 directory
                                    L5_path = os.path.join(L4_path, L5)
                                    if os.path.isdir(L5_path):
                                        for L6 in tqdm(os.listdir(L5_path), desc='Level 6'):  # Level 6 directory
                                            L6_path = os.path.join(L5_path, L6)
                                            if not os.path.isdir(L6_path):
                                                Image_File_name = L6_path
                                                Mask_File_name = Image_File_name.replace('images', 'masks').replace('\\rgb\\', '\\classSegmentation\\').replace('rgb_', 'classgt_').replace('jpg', 'png')
                                                # Adjust File name to write
                                                Name = L1 + '_' + L2 + '_' + L3 + '_' + L4 + '_' + L5 + '_'
                                                match = re.search(r'rgb_(\d+)\.jpg', Image_File_name)
                                                if match:
                                                    number_str = match.group(1)
                                                    New_ImageName = Name + number_str + '.jpg'
                                                    New_MaskName = Name + number_str + '.png'

                                                    # Copy the image file to new/images/
                                                    shutil.copy(Image_File_name, os.path.join(New_Image_folder, New_ImageName))

                                                    # Read the mask file and retain only [100, 60, 100]
                                                    mask = cv2.imread(Mask_File_name, cv2.IMREAD_COLOR)
                                                    if mask is not None:
                                                        # Create a mask to retain only [100, 60, 100]
                                                        condition = (mask == [100, 60, 100]).all(axis=2)
                                                        new_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
                                                        new_mask[condition] = 255
                                                        # Save the processed mask to new/masks/
                                                        cv2.imwrite(os.path.join(New_Mask_folder, New_MaskName), new_mask)
