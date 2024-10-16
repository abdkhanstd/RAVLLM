import os
import re
import json
import torch
import numpy as np
import supervision as sv
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoProcessor, get_scheduler
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from peft import LoraConfig, get_peft_model, PeftModel
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
import scripts.abdhead


# Set environment proxy
os.environ['http_proxy'] = 'http://x98.local:1092'
os.environ['https_proxy'] = 'http://x98.local:1092'
print("🌐 Proxy set to http://x98.local:1092")


# Constants
# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
NUM_WORKERS = 0
EPOCHS = 500
LR = 1e-6

# Paths
BEST_MODEL_DIR = 'weights/florence2/'
BEST_MODEL_PATH = os.path.join(BEST_MODEL_DIR, 'best_model.pth')
PEFT_MODEL_DIR = os.path.join(BEST_MODEL_DIR, 'peft_model')
PROCESSOR_PATH = os.path.join(BEST_MODEL_DIR, 'processor')

#CHECKPOINT = "google/paligemma-3b-mix-224"

CHECKPOINT = "microsoft/Florence-2-base-ft"
REVISION = 'refs/pr/6'

# Load or initialize the model and processor
if os.path.exists(BEST_MODEL_PATH):
    print(f"Loading the best model from {BEST_MODEL_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True).to(DEVICE)
    processor = AutoProcessor.from_pretrained(PROCESSOR_PATH,trust_remote_code=True)
    
    # Load model configuration and weights
    peft_model = PeftModel.from_pretrained(model, PEFT_MODEL_DIR, is_trainable=True)
    peft_model.load_state_dict(torch.load(BEST_MODEL_PATH,weights_only=True))  # Load weights
    peft_model.print_trainable_parameters()

else:
    print("No pre-trained best model found. Initializing model and processor from scratch.")
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True).to(DEVICE)
    processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)
    
    config = LoraConfig(
        r=64,
        lora_alpha=32,
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

# Ensure model parameters require gradients
print("Checking if all model parameters require gradients...")
for param in model.parameters():
    param.requires_grad = True

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
train_images_dir = 'datasets/LF/Train/images'
test_images_dir = 'datasets/LF/Test/images'

# Collate function for DataLoader
def collate_fn(batch):
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True)
    return inputs, answers

# Create datasets and dataloaders
train_dataset = DetectionDataset(jsonl_file_path=train_jsonl_path, image_directory_path=train_images_dir)
val_dataset = DetectionDataset(jsonl_file_path=test_jsonl_path, image_directory_path=test_images_dir)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS)

# Utility function to render images with bounding boxes
def render_image_with_bbox(image: Image.Image, detections: sv.Detections):
    image_np = np.array(image)
    box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    
    annotated_image = box_annotator.annotate(image_np, detections)
    annotated_image = label_annotator.annotate(annotated_image, detections)
    
    return Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

# Function to convert the suffix to a dictionary with bounding boxes and labels
def convert_suffix_to_dict(suffix: str) -> Dict[str, Any]:
    locs = re.findall(r'<loc_(\d+)>', suffix)
    bboxes = []
    if len(locs) % 4 == 0:
        for i in range(0, len(locs), 4):
            x1, y1, x2, y2 = map(int, locs[i:i+4])
            bboxes.append([x1, y1, x2, y2])
    data = {'<OD>': {'bboxes': bboxes, 'labels': ['object'] * len(bboxes)}}
    return data

# Function to render inference results
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

        gt_data = convert_suffix_to_dict(suffix)

        try:
            gt_detections = sv.Detections.from_lmm(
                lmm='FLORENCE_2',
                result=gt_data,
                resolution_wh=(w, h)
            )

        except Exception as e:
            print(f"Error processing ground truth data for image {data['image']}: {e}")
            gt_detections = sv.Detections.empty()

        pred_image_pil = render_image_with_bbox(image, pred_detections)
        gt_image_pil = render_image_with_bbox(oimage, gt_detections)
        print("Predicted: ", pred_detections)
        print("Ground Truth: ", gt_detections)
        
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

# Evaluation function with real-time mAP display
def evaluate_model(model, val_dataset, processor,mode='train'):
    targets = []
    predictions = []
    batch_size = 50
    num_samples = len(val_dataset.dataset)
    for i in tqdm(range(num_samples), desc="Evaluating"):
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
            image_path = os.path.join(train_images_dir if mode == 'train' else test_images_dir, data['image'])
            oimage = Image.open(image_path).convert('RGB')

            prediction = processor.post_process_generation(generated_text, task='<OD>', image_size=image.size)
            prediction = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, prediction, resolution_wh=image.size)
            if len(prediction) == 0:
                continue

            prediction.class_id = np.zeros(len(prediction.xyxy))
            prediction.confidence = np.ones(len(prediction))

            target_data = convert_suffix_to_dict(suffix)
            target = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, target_data, resolution_wh=oimage.size)
            target.class_id = np.zeros(len(target.xyxy))

            #print("GT:",target)
            #print("PD:",prediction)
            targets.append(target)
            predictions.append(prediction)

            if (i + 1) % batch_size == 0 or (i + 1) == num_samples:
                if predictions and targets:
                    try:
                        mean_average_precision = sv.MeanAveragePrecision.from_detections(
                            predictions=predictions,
                            targets=targets,
                        )
                        print("GT:",target)
                        print("PD:",prediction)
                        print(f"\nIntermediate mAP at sample {i + 1}/{num_samples}:")
                        print(f"mAP50-95: {mean_average_precision.map50_95:.2f}")
                        print(f"mAP50: {mean_average_precision.map50:.2f}")
                        print(f"mAP75: {mean_average_precision.map75:.2f}")
                    except Exception as e:
                        print(f"Error calculating mAP at sample {i + 1}: {e}")

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
        print(f"\nFinal mAP:")
        print(f"mAP50-95: {mean_average_precision.map50_95:.2f}")
        print(f"mAP50: {mean_average_precision.map50:.2f}")
        print(f"mAP75: {mean_average_precision.map75:.2f}")

    except Exception as e:
        print(f"Error calculating metrics: {e}")

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
        print(f"\nStarting Training Epoch {epoch + 1}/{epochs}")
        
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

            if not loss.requires_grad:
                raise ValueError("Loss tensor does not require gradients, check your model setup and inputs")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Average Training Loss for Epoch {epoch + 1}: {avg_train_loss}")

        model.eval()
        val_loss = 0
        print(f"Starting Validation Epoch {epoch + 1}/{epochs}")
        
        val_loss = 0
        targets = []
        predictions = []

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
                
                # Collect predictions and ground truth for mAP calculation
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3
                )
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

                # Assumption: processor.post_process_generation expects (width, height)
                image_size = (inputs["pixel_values"].shape[-1], inputs["pixel_values"].shape[-2])

                prediction = processor.post_process_generation(generated_text, task='<OD>', image_size=image_size)
                pred_detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, prediction, resolution_wh=image_size)
                
                if pred_detections:
                    predictions.append(pred_detections)

                gt_data = convert_suffix_to_dict(answers[0])  # Adjust if answers has multiple entries
                gt_detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, gt_data, resolution_wh=image_size)
                
                if gt_detections:
                    targets.append(gt_detections)

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            print(f"Average Validation Loss for Epoch {epoch + 1}: {avg_val_loss}")



            scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            os.makedirs(BEST_MODEL_DIR, exist_ok=True)
            peft_model.save_pretrained(PEFT_MODEL_DIR, save_embedding_layers=True,save_adapters=True)
            torch.save(peft_model.state_dict(), BEST_MODEL_PATH)  # Save state dictionary
            processor.save_pretrained(PROCESSOR_PATH)  # Save processor

            print(f"New best model saved with validation loss: {best_val_loss}")
        else:
            patience_counter += 1
            print(f"No improvement in validation loss. Patience counter: {patience_counter}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        print(f"\nSample predictions and ground truth for Epoch {epoch + 1}:")
        render_inference_results(model, val_loader.dataset, 25, mode='val')
        evaluate_model(peft_model, val_dataset, processor,mode='test')


# Run the training and evaluation
train_model(train_loader, val_loader, peft_model, processor)


# Final evaluation after training
print("\nFinal evaluation on validation dataset:")
evaluate_model(peft_model, val_dataset, processor,mode='test')
