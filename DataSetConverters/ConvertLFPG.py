#Convert Lost and Found dataset to PaliGemma Orientation

import os
import re
import json
import torch
import numpy as np
import supervision as sv
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, AutoModelForCausalLM, AutoProcessor, get_scheduler
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Constants
CHECKPOINT = "microsoft/Florence-2-large-ft"
REVISION = 'refs/pr/6'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
NUM_WORKERS = 0
EPOCHS = 500
LR = 1e-6

# Load model and processor
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True).to(DEVICE)
processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)

# Check if the best model exists and load it
if os.path.exists('best_model.pth'):
    print("Loading the best model from checkpoint...")
    model.load_state_dict(torch.load('best_model.pth'))

# Dataset classes
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
            image = Image.open(image_path).convert('RGB')
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

# Data paths
train_jsonl_path = 'datasets/LF/Train/train.jsonl'
test_jsonl_path = 'datasets/LF/Test/test.jsonl'
train_images_dir = 'datasets/LF/Train'
test_images_dir = 'datasets/LF/Test'

# Collate function
def collate_fn(batch):
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True)
    return inputs, answers

# Create datasets and dataloaders
train_dataset = DetectionDataset(jsonl_file_path=train_jsonl_path, image_directory_path=train_images_dir)
val_dataset = DetectionDataset(jsonl_file_path=test_jsonl_path, image_directory_path=test_images_dir)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS)

# LoRA configuration
config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "linear", "Conv2d", "lm_head", "fc2"],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
    bias="none",
    inference_mode=False,
    use_rslora=True,
    init_lora_weights="gaussian",
    revision=REVISION
)

peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()

# Utility functions
def render_image_with_bbox(image: Image.Image, detections: sv.Detections):
    image_np = np.array(image)
    # Create annotators
    box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    
    # Annotate image
    annotated_image = box_annotator.annotate(image_np, detections)
    annotated_image = label_annotator.annotate(annotated_image, detections)
    
    return Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

def convert_suffix_to_dict(suffix: str) -> str:
    # Extract bounding box coordinates using regex
    locs = re.findall(r'<loc_(\d+)>', suffix)
    bboxes = []
    if len(locs) % 4 == 0:
        for i in range(0, len(locs), 4):
            x1, y1, x2, y2 = map(int, locs[i:i+4])
            bboxes.append([x1, y1, x2, y2])
    data = {'<OD>': {'bboxes': bboxes, 'labels': ['object'] * len(bboxes)}}

    # Convert the dictionary to a JSON string to ensure double quotes
    data_json = json.dumps(data)

    # Alternatively, if still seeing single quotes, manually replace
    data_json = data_json.replace("'", '"')
    data_json = data_json.replace('"object"', "'object'")

    return data

def render_inference_results(model, dataset: DetectionDataset, count: int, mode='train'):
    count = min(count, len(dataset))
    for i in range(count):
        image, data = dataset.dataset[i]
        prefix = data['prefix']
        suffix = data['suffix']
        image_path = os.path.join(train_images_dir if mode == 'train' else test_images_dir, data['image'])
        oimage = Image.open(image_path).convert('RGB')
        w, h = oimage.size

        # Model prediction
        inputs = processor(text=prefix, images=image, return_tensors="pt").to(DEVICE)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        prediction = processor.post_process_generation(generated_text, task='<OD>', image_size=image.size)
        pred_detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, prediction, resolution_wh=image.size)

        # Convert suffix to required format
        gt_data = convert_suffix_to_dict(suffix)

        # Debug: Print GT data and suffix
        print("GT data:", gt_data)
        print("GT suffix:", suffix)

        try:
            gt_detections = sv.Detections.from_lmm(
                lmm='FLORENCE_2',
                result=gt_data,
                resolution_wh=(w, h)
            )

            # Debug: Print GT detections
            print("GT detections:", gt_detections)

            # Check if GT detections are empty
            if gt_detections.xyxy.size == 0:
                print("No valid GT detections found.")

        except Exception as e:
            print(f"Error processing GT data for image {data['image']}: {e}")
            gt_detections = sv.Detections.empty()

        # Render images
        pred_image_pil = render_image_with_bbox(image, pred_detections)
        gt_image_pil = render_image_with_bbox(oimage, gt_detections)
        
        # Display images
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(pred_image_pil)
        plt.title(f"Model Prediction {i+1}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(gt_image_pil)
        plt.title(f"Ground Truth {i+1}")
        plt.axis('off')

        plt.show()


# Training function
def train_model(train_loader, val_loader, model, processor, epochs=EPOCHS, lr=LR):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, answers in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False
            ).input_ids.to(DEVICE)

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Average Training Loss: {avg_train_loss}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, answers in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                labels = processor.tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False
                ).input_ids.to(DEVICE)

                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            print(f"Average Validation Loss: {avg_val_loss}")

            scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        output_dir = f"./model_checkpoints/epoch_{epoch+1}"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)

        print(f"\nSample predictions and ground truth for Epoch {epoch + 1}:")
        render_inference_results(model, val_loader.dataset, 6, mode='val')

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses')
    plt.savefig('loss_plot.png')
    plt.show()

# Evaluation function
def extract_classes(dataset: DetectionDataset):
    PATTERN = r'([a-zA-Z0-9 ]+ of [a-zA-Z0-9 ]+)<loc_\d+>'
    class_set = set()
    for i in range(len(dataset.dataset)):
        image, data = dataset.dataset[i]
        suffix = data["suffix"]
        classes = re.findall(PATTERN, suffix)
        class_set.update(classes)
    return sorted(class_set)

def evaluate_model(model, val_dataset, processor):
    CLASSES = extract_classes(train_dataset)
    if not CLASSES:
        print("Warning: CLASSES list is empty")
        return

    targets = []
    predictions = []
    for i in range(len(val_dataset.dataset)):
        try:
            image, data = val_dataset.dataset[i]
            prefix = data['prefix']
            suffix = data['suffix']

            inputs = processor(text=prefix, images=image, return_tensors="pt").to(DEVICE)
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

            prediction = processor.post_process_generation(generated_text, task='<OD>', image_size=image.size)
            prediction = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, prediction, resolution_wh=image.size)
            if len(prediction) == 0:
                continue

            valid_predictions = np.isin(prediction.class_name, CLASSES)
            if not np.any(valid_predictions):
                continue

            prediction = prediction[valid_predictions]
            prediction.class_id = np.array([CLASSES.index(class_name) for class_name in prediction.class_name])
            prediction.confidence = np.ones(len(prediction))

            target = processor.post_process_generation(suffix, task='<OD>', image_size=image.size)
            target = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, target, resolution_wh=image.size)
            target.class_id = np.array([CLASSES.index(class_name) for class_name in target.class_name])

            targets.append(target)
            predictions.append(prediction)

        except Exception as e:
            print(f"Error processing image {i}: {e}")
            continue

    if not predictions or not targets:
        print("Error: No valid predictions or targets generated")
        return

    try:
        mean_average_precision = sv.MeanAveragePrecision.from_detections(
            predictions=predictions,
            targets=targets,
        )

        print(f"mAP50-95: {mean_average_precision.map50_95:.2f}")
        print(f"mAP50: {mean_average_precision.map50:.2f}")
        print(f"mAP75: {mean_average_precision.map75:.2f}")

        # Calculate overall precision, recall, and F1 score
        precision, recall, f1 = sv.precision_recall_f1(
            predictions=predictions, targets=targets
        )
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")

    except Exception as e:
        print(f"Error calculating metrics: {e}")

# Run the training
# train_model(train_loader, val_loader, peft_model, processor)

# Evaluate the model
evaluate_model(peft_model, val_dataset, processor)
