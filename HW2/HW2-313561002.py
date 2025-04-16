import os
os.environ['TORCH_HOME'] = os.getcwd()
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import time
import json
from torch.optim.lr_scheduler import StepLR
import cv2
from tqdm import tqdm
import random
from torchvision.ops import nms
import math
from PIL import ImageDraw

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"using device: {device}")
print(torch.cuda.get_device_name(0))
print(os.getcwd())

# Directories
DATA_ROOT = "/home/nsslab/Desktop/Patrick/Data/nycu-hw2-data"
TRAIN_JSON = os.path.join(DATA_ROOT, "train.json")
VAL_JSON = os.path.join(DATA_ROOT, "valid.json")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train")
VAL_IMG_DIR = os.path.join(DATA_ROOT, "valid")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test")
OUTPUT_DIR = "output"

# Training parameters
BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
NUM_WORKERS = 2
NUM_CLASSES = 10

# Checkpoint parameters
SAVE_CHECKPOINT_EVERY = 2  # Save a checkpoint every N epochs
CHECKPOINT_PREFIX = "faster_rcnn_checkpoint_epoch_"

# Dataset class
class DigitDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, coco_json, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        self.coco = COCO(coco_json)
        self.image_ids = list(sorted(self.coco.imgs.keys()))
        
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_filename = f"{img_id}.png"
        img_path = os.path.join(self.img_dir, img_filename)
        
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: {img_path} not found.")
            img_files = os.listdir(self.img_dir)
            if len(img_files) > idx:
                alt_img_path = os.path.join(self.img_dir, img_files[idx])
                img = Image.open(alt_img_path).convert("RGB")
                print(f"Using alternative file: {alt_img_path}")
            else:
                raise FileNotFoundError(f"No suitable image file found for index {idx}")
        
        # Get all annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes = []
        labels = []
        
        for ann in anns:
            x_min, y_min, width, height = ann['bbox']
            x_max = x_min + width
            y_max = y_min + height
            
            # Ensure box coordinates are valid
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img.width, x_max)
            y_max = min(img.height, y_max)
            
            if x_max <= x_min or y_max <= y_min:
                continue
                
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann['category_id'])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target
    
    def __len__(self):
        return len(self.image_ids)

# Data transformation
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            
        return image, target

class RandomBrightness:
    def __init__(self, brightness_factor_range=(0.8, 1.2)):
        self.brightness_factor_range = brightness_factor_range
    
    def __call__(self, image, target):
        factor = random.uniform(*self.brightness_factor_range)
        image = F.adjust_brightness(image, factor)
        return image, target

class RandomContrast:
    def __init__(self, contrast_factor_range=(0.8, 1.2)):
        self.contrast_factor_range = contrast_factor_range
    
    def __call__(self, image, target):
        factor = random.uniform(*self.contrast_factor_range)
        image = F.adjust_contrast(image, factor)
        return image, target

# Create data transformations
def get_transform(train):
    transforms = [ToTensor()]
    
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
        transforms.append(RandomBrightness())
        transforms.append(RandomContrast())
    
    transforms.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    return Compose(transforms)

# Create model
def get_faster_rcnn_model(num_classes):
    backbone = resnet_fpn_backbone('resnet101', pretrained=True)
    
    # Customize RPN
    anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    
    # Create Faster R-CNN model
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes + 1,  # +1 for background
        rpn_anchor_generator=anchor_generator,
        min_size=600,
        max_size=1000,
        box_score_thresh=0.05,
        box_nms_thresh=0.5
    )
    
    return model

# Collate function for data loader
def collate_fn(batch):
    return tuple(zip(*batch))

# Post-processing for Task 2: Convert detected digits to whole numbers
def predict_whole_number(image_id, detections, conf_threshold=0.5, max_digits=4, iou_threshold=0.4):
    if not isinstance(detections, list) or len(detections) == 0:
        return -1

    relevant_detections = [d for d in detections if d['image_id'] == image_id and d['score'] >= conf_threshold]
    if len(relevant_detections) == 0:
        return -1
    
    # Convert bounding boxes for NMS
    boxes = []
    scores = []
    labels = []
    detection_mapping = []
    
    for i, det in enumerate(relevant_detections):
        x, y, w, h = det['bbox']
        boxes.append([x, y, x+w, y+h])
        scores.append(det['score'])
        labels.append(det['category_id'])
        detection_mapping.append(i)
    
    if not boxes:
        return -1
    
    # Apply NMS
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold)
    
    # Filtered detections after NMS
    filtered_detections = []
    for idx in keep_indices:
        idx = idx.item()
        original_idx = detection_mapping[idx]
        det = relevant_detections[original_idx]
        
        # Ensure category_id digit correctness
        category_id = det['category_id']
        if category_id < 1 or category_id > 10:
            continue
        
        filtered_detections.append({
            'image_id': image_id,
            'category_id': category_id,
            'bbox': det['bbox'],
            'score': det['score']
        })
    
    if len(filtered_detections) == 0:
        return -1
        
    if len(filtered_detections) > max_digits:
        filtered_detections.sort(key=lambda x: x['score'], reverse=True)
        filtered_detections = filtered_detections[:max_digits]
    
    # Ensure correct digit sort
    filtered_detections.sort(key=lambda x: x['bbox'][0])
    
    digits = []
    for det in filtered_detections:
        digit = (det['category_id']) % 10
        digits.append(str(digit))
    if not digits:
        return -1

    # Combine digits to form the whole number
    whole_number = int(''.join(digits))
    return whole_number

# Training
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    
    total_loss = 0
    batch_losses = []
    
    for images, targets in tqdm(data_loader, desc="Training"):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        batch_losses.append(losses.item())
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss, batch_losses

# Evaluation
def evaluate(model, data_loader, device):
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = list(img.to(device) for img in images)
            outputs = model(images)
            
            for i, (output, target) in enumerate(zip(outputs, targets)):
                image_id = target["image_id"].item()
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                
                for box, score, label in zip(boxes, scores, labels):
                    if score > 0.05:  # Minimum score threshold
                        x1, y1, x2, y2 = box
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Convert box format from [x1, y1, x2, y2] to COCO format [x1, y1, width, height]
                        coco_box = [float(x1), float(y1), float(width), float(height)]
                        
                        prediction = {
                            "image_id": image_id,
                            "category_id": int(label),
                            "bbox": coco_box,
                            "score": float(score)
                        }
                        all_predictions.append(prediction)

    return all_predictions

# Calculate mAP using pycocotools
def calculate_map(gt_json, predictions_file=None, predictions=None):
    if predictions_file is None and predictions is None:
        raise ValueError("Either predictions_file or predictions must be provided")
    
    coco_gt = COCO(gt_json)
    
    if predictions_file:
        if isinstance(predictions_file, str):
            coco_dt = coco_gt.loadRes(predictions_file)
        else:
            coco_dt = coco_gt.loadRes(predictions)
    else:
        # Save predictions to a temporary file
        temp_file = os.path.join(OUTPUT_DIR, "temp_predictions.json")
        with open(temp_file, 'w') as f:
            json.dump(predictions, f)
        coco_dt = coco_gt.loadRes(temp_file)
        os.remove(temp_file)  # Clean up
    
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    return coco_eval.stats[0]

# Create Task 2 ground truth from a COCO JSON file
def create_task2_ground_truth(coco_json, img_dir, output_csv):
    coco_gt = COCO(coco_json)
    image_ids = list(sorted(coco_gt.imgs.keys()))
    task2_labels = []
    
    for img_id in image_ids:
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        
        if len(anns) == 0:
            task2_labels.append({"image_id": img_id, "label": -1})
            continue
        
        anns.sort(key=lambda x: x['bbox'][0])
        digits = [str(ann['category_id']) for ann in anns]
        whole_number = int(''.join(digits))
        
        task2_labels.append({"image_id": img_id, "label": whole_number})
    
    # Create and save the DataFrame
    task2_df = pd.DataFrame(task2_labels)
    task2_df.to_csv(output_csv, index=False)
    print(f"Task 2 ground truth saved to {output_csv}")
    
    return task2_df

# Evaluate Task 2
def evaluate_task2(gt_csv, pred_csv):
    gt_df = pd.read_csv(gt_csv)
    pred_df = pd.read_csv(pred_csv)

    merged = pd.merge(gt_df, pred_df, on='image_id', how='left')
    accuracy = (merged['label'] == merged['pred_label']).mean()
    print(f"Task 2 Accuracy: {accuracy:.4f}")
    
    return accuracy

# Function to visualize some sample predictions
def visualize_predictions(img_dir, predictions, num_samples=5, output_dir=None):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    pred_by_img = {}
    for pred in predictions:
        img_id = pred['image_id']
        if img_id not in pred_by_img:
            pred_by_img[img_id] = []
        pred_by_img[img_id].append(pred)

    sample_ids = random.sample(list(pred_by_img.keys()), min(num_samples, len(pred_by_img)))
    
    for img_id in sample_ids:
        img_path = os.path.join(img_dir, f"{img_id}.png")
        if not os.path.exists(img_path):
            for file in os.listdir(img_dir):
                if file.startswith(f"{img_id}.") or file == f"{img_id}":
                    img_path = os.path.join(img_dir, file)
                    break
        
        if not os.path.exists(img_path):
            print(f"Image file for ID {img_id} not found.")
            continue
   
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        preds = pred_by_img[img_id]
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                 (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0), 
                 (0, 0, 128), (128, 128, 0)]
        
        for pred in preds:
            x, y, w, h = pred['bbox']
            category_id = pred['category_id']
            score = pred['score']
            color = colors[category_id % len(colors)]
            draw.rectangle([x, y, x+w, y+h], outline=color, width=2)
            draw.text((x, y-15), f"{category_id} ({score:.2f})", fill=color)
        
        whole_number = predict_whole_number(img_id, preds, conf_threshold=0.5, max_digits=4, iou_threshold=0.4)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(np.array(img))
        plt.title(f"Image ID: {img_id}, Predicted Number: {whole_number}")
        plt.axis('off')
        
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, f"pred_{img_id}.png"))
            plt.close()
        else:
            plt.show()

# Function to fix category IDs with careful handling of -1
def fix_pred_label(label):
    if label == "-1":
        return -1

    digits = str(label)
    corrected_digits = []
    for digit in digits:
        if digit.isdigit():
            digit_val = int(digit)
            # Change from category_id to actual digit
            actual_digit = digit_val - 1
            if actual_digit < 0:
                actual_digit = 9
            corrected_digits.append(str(actual_digit))

    if not corrected_digits:
        return label

    return int(''.join(corrected_digits))

def main():
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create datasets
    train_dataset = DigitDetectionDataset(TRAIN_IMG_DIR, TRAIN_JSON, transforms=get_transform(train=True))
    val_dataset = DigitDetectionDataset(VAL_IMG_DIR, VAL_JSON, transforms=get_transform(train=False))
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn
    )
    
    # Create model
    model = get_faster_rcnn_model(NUM_CLASSES)
    model.to(device)
    
    # Print model summary
    print(f"Model created: Faster R-CNN with ResNet101-FPN backbone")
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Initialize tracking variables
    best_map = 0
    training_losses = []
    validation_maps = []
    
    # Create ground truth for Task 2 validtaion
    val_task2_csv = os.path.join(OUTPUT_DIR, "val_task2_gt.csv")
    create_task2_ground_truth(VAL_JSON, VAL_IMG_DIR, val_task2_csv)
    
    # Training loop
    print(f"Start training for {NUM_EPOCHS} epochs: ")
    
    # Use checkpoint system
    checkpoints = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(CHECKPOINT_PREFIX) and f.endswith('.pth')]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint_path = os.path.join(OUTPUT_DIR, latest_checkpoint)
        start_epoch = int(latest_checkpoint.split('_')[-1].split('.')[0])
        print(f"Resuming from checkpoint {checkpoint_path}, starting at epoch {start_epoch+1}")
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        start_epoch = 0
    
    #start_epoch = 0
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        start_time = time.time()
        
        # Train
        train_loss, batch_losses = train_one_epoch(model, optimizer, train_loader, device)
        training_losses.append(train_loss)
        
        print(f"Training loss: {train_loss:.4f}")
        
        # Update learning rate
        lr_scheduler.step()
        
        # Save checkpoints
        if (epoch + 1) % SAVE_CHECKPOINT_EVERY == 0:
            checkpoint_path = os.path.join(OUTPUT_DIR, f"{CHECKPOINT_PREFIX}{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Evaluate model
        print("Evaluating model")
        predictions = evaluate(model, val_loader, device)
        
        # Save predictions to JSON file
        predictions_file = os.path.join(OUTPUT_DIR, f"val_predictions_epoch_{epoch+1}.json")
        with open(predictions_file, 'w') as f:
            json.dump(predictions, f)
        
        # Calculate mAP
        print("Calculating mAP: ")
        map_score = calculate_map(VAL_JSON, predictions_file)
        validation_maps.append(map_score)
        print(f"mAP: {map_score:.4f}")
        
        # Evaluate task 2
        print("Evaluating Task 2")
        task2_predictions = []
        image_ids = list(set([p['image_id'] for p in predictions]))
        for img_id in image_ids:
            img_preds = [p for p in predictions if p['image_id'] == img_id]
            whole_number = predict_whole_number(img_id, img_preds, conf_threshold=0.5, max_digits=4, iou_threshold=0.4)
            task2_predictions.append({"image_id": img_id, "pred_label": whole_number})
        
        # Save Task 2 predictions
        task2_pred_csv = os.path.join(OUTPUT_DIR, f"val_task2_pred_epoch_{epoch+1}.csv")
        pd.DataFrame(task2_predictions).to_csv(task2_pred_csv, index=False)

        # Task 2 accuracy
        task2_accuracy = evaluate_task2(val_task2_csv, task2_pred_csv)
        
        # Save model if better
        if map_score > best_map:
            best_map = map_score
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            print(f"Saved new best model with mAP: {best_map:.4f}")
        
        # Visualize some predictions
        if (epoch + 1) % 5 == 0:
            visualize_dir = os.path.join(OUTPUT_DIR, f"visualizations_epoch_{epoch+1}")
            visualize_predictions(VAL_IMG_DIR, predictions, num_samples=5, output_dir=visualize_dir)
        
        epoch_time = time.time() - start_time
        print(f"Epoch completed in {epoch_time:.2f} seconds")
        
        # Print summary of this epoch
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Training Loss: {train_loss:.4f}")
        print(f"  Validation mAP: {map_score:.4f}")
        print(f"  Task 2 Accuracy: {task2_accuracy:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    print("\nTraining completed")
    
    # Plot training and validation metrics
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, NUM_EPOCHS + 1), training_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, NUM_EPOCHS + 1), validation_maps)
    plt.title('Validation mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_metrics.png'))
    plt.close()
    
    print("\nGenerating test predictions using best model...")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth")))
    model.eval()

    test_predictions_task1 = []
    test_predictions_task2 = []
    test_image_files = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith('.png')]
    
    # Process each test image
    for img_file in tqdm(test_image_files, desc="Processing test images"):
        img_path = os.path.join(TEST_IMG_DIR, img_file)
        img = Image.open(img_path).convert("RGB")
        img_tensor = F.normalize(F.to_tensor(img), 
                                mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])

        image_id = int(img_file.split('.')[0])
        with torch.no_grad():
            prediction = model([img_tensor.to(device)])

        boxes = prediction[0]['boxes'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()
        labels = prediction[0]['labels'].cpu().numpy()
        image_predictions = []
        for box, score, label in zip(boxes, scores, labels):
            if score > 0.05:
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                coco_box = [float(x1), float(y1), float(width), float(height)]
                
                # Task 1 Submission File
                pred = {
                    "image_id": image_id,
                    "bbox": coco_box,
                    "score": float(score),
                    "category_id": int(label)
                }
                image_predictions.append(pred)
                test_predictions_task1.append(pred)
        
        # Task 2 Submission File
        whole_number = predict_whole_number(image_id, image_predictions, conf_threshold=0.5, max_digits=4, iou_threshold=0.4)
        test_predictions_task2.append({"image_id": image_id, "pred_label": whole_number})
    
    # Save Predictionv Files
    with open(os.path.join(OUTPUT_DIR, "pred.json"), 'w') as f:
        json.dump(test_predictions_task1, f)
    task2_df = pd.DataFrame(test_predictions_task2)
    task2_df.to_csv(os.path.join(OUTPUT_DIR, "pred.csv"), index=False)
    
    print(f"\nPredictions saved!")
    print(f"Task 1 predictions: {os.path.join(OUTPUT_DIR, 'pred.json')}")
    print(f"Task 2 predictions: {os.path.join(OUTPUT_DIR, 'pred.csv')}")

    pred_df = pd.read_csv("output/pred.csv")
    print(f"Loaded {len(pred_df)} predictions")
    corrected_predictions = []

    for _, row in pred_df.iterrows():
        image_id = row['image_id']
        pred = row['pred_label']
        
        # Explicit handling of different types
        if pd.isna(pred):
            corrected_pred = -1
        elif pred == -1:
            corrected_pred = -1
        else:
            corrected_pred = fix_pred_label(pred)
            
        corrected_predictions.append({
            'image_id': image_id,
            'pred_label': corrected_pred
        })

    corrected_df = pd.DataFrame(corrected_predictions)
    print(corrected_df.head(10))

    # Save Fixed predictions
    corrected_df.to_csv("output/pred_fixed.csv", index=False)
    print("\nFixed predictions saved to output/pred_fixed.csv")

    # Sort Prediction According to image_id
    df = pd.read_csv("./output/pred_fixed.csv")
    df['image_id'] = df['image_id'].astype(int)  # Only if image_id is not already int
    df_sorted = df.sort_values(by='image_id')
    df_sorted = df_sorted.reset_index(drop=True)
    df_sorted.to_csv("pred_sort.csv", index=False)

if __name__ == "__main__":
    main()