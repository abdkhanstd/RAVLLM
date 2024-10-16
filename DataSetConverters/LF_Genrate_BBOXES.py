import os
import cv2
import json
import numpy as np

def generate_index_json(base_path):
    images_path = os.path.join(base_path, 'images')
    masks_path = os.path.join(base_path, 'masks')

    for mode in ['train', 'test']:
        img_dir = os.path.join(images_path, mode)
        mask_dir = os.path.join(masks_path, mode)
        
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
                    for i, contour in enumerate(contours):
                        x, y, w, h = cv2.boundingRect(contour)
                        roi_rect = [[x, y], [x+w, y+h]]
                        crop_rect = roi_rect  # Adjust this based on your needs
                        instances.append({
                            "id": i+1,
                            "cls": 2,
                            "crop_rect": crop_rect,
                            "roi_rect": roi_rect
                        })

                    # Debugging information
                    if len(instances) == 0:
                        print(f"No instances found in mask: {mask_path}")
                    else:
                        print(f"Found {len(instances)} instances in mask: {mask_path}")
                    
                    annotation = {"instances": instances}
                    
                    json_path = os.path.join(mask_subdir, img_name.replace('leftImg8bit_', '').replace('.png', '_index.json'))
                    with open(json_path, 'w') as json_file:
                        json.dump(annotation, json_file, indent=4)
                    print(f"Generated {json_path}")

# Base path
base_path = 'RA2'
generate_index_json(base_path)
