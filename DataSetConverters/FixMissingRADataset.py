import os
import json
import cv2
import numpy as np

def generate_annotations(mask_image_path, class_color):
    """
    Generate annotations for a given mask image.

    Args:
        mask_image_path (str): Path to the mask image.
        class_color (list): BGR color value for the class of interest.

    Returns:
        list: List of dictionaries containing annotations.
    """
    mask_image = cv2.imread(mask_image_path)
    annotations = []
    height, width, _ = mask_image.shape

    binary_mask = cv2.inRange(mask_image, np.array(class_color), np.array(class_color))

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        annotation = {
            'id': i + 1,  # Assign a unique id starting from 1
            'cls': 2,  # Assuming 'cls' is a fixed class id (modify as needed)
            'crop_rect': [[x, y], [x + w, y + h]],
            'roi_rect': [[x, y], [x + w, y + h]]
        }
        annotations.append(annotation)

    return annotations

def process_folder(images_path, masks_path, class_color):
    """
    Process all images and masks in the given folder to generate annotations.

    Args:
        images_path (str): Path to the folder containing images.
        masks_path (str): Path to the folder containing masks.
        class_color (list): BGR color value for the class of interest.
    """
    for root, _, files in os.walk(images_path):
        for file in files:
            if file.endswith('.jpg'):
                image_name = file
                img_name = file.replace('.jpg', '')
                mask_folder = os.path.join(masks_path, img_name + ".labels")
                mask_path = os.path.join(mask_folder, "labels_semantic_color.png")

                if not os.path.exists(mask_path):
                    print(f"Mask {mask_path} does not exist. Skipping.")
                    continue

                annotations = generate_annotations(mask_path, class_color)

                index_json_path = os.path.join(mask_folder, 'index_.json')
                with open(index_json_path, 'w') as f:
                    json.dump({'instances': annotations}, f, indent=4)

                print(f'Generated {index_json_path} for image {image_name}')

def main():
    images_path = 'datasets/RA/images'
    masks_path = 'datasets/RA/masks'
    class_color = [0, 69, 255]  # Example class color (Orange) in BGR format

    process_folder(images_path, masks_path, class_color)

if __name__ == "__main__":
    main()
