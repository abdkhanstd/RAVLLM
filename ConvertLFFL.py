import os
import cv2
import json
import numpy as np

def normalize_coordinates(x, y, original_width, original_height, target_width, target_height):
    """Normalize coordinates to the target size."""
    x_norm = int(x / original_width * target_width)
    y_norm = int(y / original_height * target_height)
    return x_norm, y_norm

def generate_index_json(base_path, output_base, target_size=(1000, 1000)):
    images_path = os.path.join(base_path, 'images')
    masks_path = os.path.join(base_path, 'masks')

    for mode in ['train', 'test']:
        img_dir = os.path.join(images_path, mode)
        mask_dir = os.path.join(masks_path, mode)
        output_img_dir = os.path.join(output_base, mode.capitalize(), 'images')
        output_mask_dir = os.path.join(output_base, mode.capitalize(), 'masks')
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_mask_dir, exist_ok=True)

        jsonl_filename = f'{mode}.jsonl'
        jsonl_path = os.path.join(output_base, mode.capitalize(), jsonl_filename)

        with open(jsonl_path, 'w') as jsonl_file:
            for subfolder in os.listdir(img_dir):
                img_subdir = os.path.join(img_dir, subfolder)
                mask_subdir = os.path.join(mask_dir, subfolder)
                if not os.path.isdir(img_subdir):
                    continue

                for file in os.listdir(img_subdir):
                    if file.endswith('.png'):
                        img_path = os.path.join(img_subdir, file)
                        mask_name = file.replace('leftImg8bit_', '').replace('.png', '_gtCoarse_color.png')
                        mask_path = os.path.join(mask_subdir, mask_name).replace('_leftImg8bit', '')
                        json_polygon_path = mask_path.replace('_gtCoarse_color.png', '_gtCoarse_polygons.json')

                        if not os.path.exists(mask_path):
                            continue

                        # Load and resize the image
                        image = cv2.imread(img_path)
                        original_height, original_width = image.shape[:2]
                        resized_image = cv2.resize(image, target_size)
                        new_img_name = os.path.basename(img_path)
                        resized_img_path = os.path.join(output_img_dir, new_img_name)
                        cv2.imwrite(resized_img_path, resized_image)

                        suffix = ""
                        mask = cv2.imread(mask_path)
                        if mask is not None:
                            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                            target_color = [0, 0, 142]  # RGB color to detect
                            class_name = "object"

                            # Create a binary mask for the target color
                            mask_bin = np.all(mask_rgb == target_color, axis=-1).astype(np.uint8)
                            if np.any(mask_bin):  # Check if any target color is found
                                resized_mask_bin = cv2.resize(mask_bin, target_size, interpolation=cv2.INTER_NEAREST)
                                resized_mask_color = np.zeros_like(resized_mask_bin, dtype=np.uint8)
                                resized_mask_color[resized_mask_bin == 1] = 255  # Set target color to white

                                new_mask_name = os.path.basename(mask_path)
                                resized_mask_path = os.path.join(output_mask_dir, new_mask_name)
                                cv2.imwrite(resized_mask_path, resized_mask_color)

                                # Find all bounding boxes in the resized mask
                                contours, _ = cv2.findContours(resized_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                for contour in contours:
                                    x, y, w, h = cv2.boundingRect(contour)
                                    x2, y2 = x + w, y + h

                                    suffix += f'{class_name}<loc_{x}><loc_{y}><loc_{x2}><loc_{y2}>'
                        else:
                            # Use polygon information if mask is all black
                            if os.path.exists(json_polygon_path):
                                with open(json_polygon_path, 'r') as f:
                                    polygons = json.load(f)
                                    class_name = "object"

                                    for obj in polygons['objects']:
                                        if obj['label'] == 'out of roi':
                                            continue
                                        polygon = obj['polygon']
                                        x_coords = [p[0] for p in polygon]
                                        y_coords = [p[1] for p in polygon]
                                        x, y, w, h = min(x_coords), min(y_coords), max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)
                                        x2, y2 = x + w, y + h

                                        # Normalize coordinates to the target size
                                        x_norm, y_norm = normalize_coordinates(x, y, original_width, original_height, *target_size)
                                        x2_norm, y2_norm = normalize_coordinates(x2, y2, original_width, original_height, *target_size)

                                        suffix += f'{class_name}<loc_{x_norm}><loc_{y_norm}><loc_{x2_norm}><loc_{y2_norm}>'

                        # Write JSONL entry
                        if suffix:  # Only write if suffix is not empty
                            jsonl_entry = {
                                "image": new_img_name,
                                "prefix": "<OD>",
                                "suffix": suffix
                            }
                            jsonl_file.write(json.dumps(jsonl_entry) + "\n")
                            print(f"Processed {img_path}")
                        else:
                            print(f"No valid objects found for {img_path}")

# Base path and output path
base_path = 'LostAndFound'
output_base = 'LF'
generate_index_json(base_path, output_base)
