import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import cv2
import skimage.io as sio
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from utils import encode_mask, decode_maskobj
from tqdm.auto import tqdm
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import shutil
import zipfile

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"using device: {device}")
print(torch.cuda.get_device_name(0))

DATA_ROOT = "/home/nsslab/Desktop/Patrick/Data"
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_release")
OUTPUT_DIR = "output"

with open(os.path.join(DATA_ROOT,"test_image_name_to_ids.json"), 'r') as f:
    test_image_to_id = json.load(f)

CLASS_NAMES = ['background', 'class1', 'class2', 'class3', 'class4']
CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

def extract_instances(mask_path):
    mask = sio.imread(mask_path)
    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids > 0]
    
    instances = []
    for instance_id in instance_ids:
        binary_mask = (mask == instance_id).astype(np.uint8)
        
        # Get bounding box
        pos = np.where(binary_mask)
        if len(pos[0]) == 0: 
            continue
            
        # Get coordinates
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        
        width = xmax - xmin
        height = ymax - ymin
        
        if width <= 0 or height <= 0:
            continue
            
        # Only add if the instance has a reasonable size
        if width > 5 and height > 5:
            class_id = int(mask_path.stem[-1]) if mask_path.stem[-1].isdigit() else 1
            instances.append({
                'mask': binary_mask,
                'bbox': [xmin, ymin, xmax, ymax],
                'class_id': class_id
            })
    
    return instances

class CellSegmentationDataset(Dataset):
    def __init__(self, image_dirs, transform=None, train=True):
        self.transform = transform
        self.train = train
        self.image_paths = []
        self.image_dirs = [Path(img_dir) if isinstance(img_dir, str) else img_dir 
                          for img_dir in image_dirs]
        
        # Collect image paths
        for img_dir in self.image_dirs:
            if train: # Train data
                self.image_paths.extend(list(img_dir.glob('*/image.tif')))
            else: # Test data
                self.image_paths.extend(list(img_dir.glob('*.tif')))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.train: # For training data
            parent_dir = img_path.parent
            mask_files = list(parent_dir.glob('class*.tif'))
            
            all_instances = []
            for mask_file in mask_files:
                class_instances = extract_instances(mask_file)
                all_instances.extend(class_instances)
            
            masks = []
            boxes = []
            labels = []
            
            for instance in all_instances:
                masks.append(instance['mask'])
                boxes.append(instance['bbox'])
                labels.append(instance['class_id'])
            
            # Skip images with no instances
            if not masks:
                dummy_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                dummy_box = [0, 0, 10, 10]
                dummy_label = 1
                
                masks = [dummy_mask]
                boxes = [dummy_box]
                labels = [dummy_label]
            
            # Apply transformations
            if self.transform:
                transformed = self.transform(image=image, masks=masks, bboxes=boxes, class_labels=labels)
                image = transformed['image']
                masks = transformed['masks']
                boxes = transformed['bboxes']
                labels = transformed['class_labels']
            
            # Convert to tensors
            if masks:
                masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)
                
                target = {
                    'boxes': boxes,
                    'labels': labels,
                    'masks': masks,
                    'image_id': torch.tensor([idx]),
                    'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
                    'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
                }
            else:
                # Handle images with no instances after augmentation
                target = {
                    'boxes': torch.zeros((0, 4), dtype=torch.float32),
                    'labels': torch.zeros((0,), dtype=torch.int64),
                    'masks': torch.zeros((0, image.shape[1], image.shape[2]), dtype=torch.uint8),
                    'image_id': torch.tensor([idx]),
                    'area': torch.zeros((0,), dtype=torch.float32),
                    'iscrowd': torch.zeros((0,), dtype=torch.int64)
                }
            
            return image, target, str(img_path)
        else: # For test data
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            
            return image, str(img_path)

# Define transformations for training
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(p=0.5, shift_limit=0.0625, scale_limit=0.1, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT),
    A.OneOf([
        A.GaussNoise(p=1),
        A.GaussianBlur(p=1),
        A.MotionBlur(p=1),
    ], p=0.2),
    A.OneOf([
        A.RandomBrightnessContrast(p=1),
        A.HueSaturationValue(p=1),
        A.RGBShift(p=1),
    ], p=0.3),
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5),
    ], p=0.2),
    A.Normalize(),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Define transformations for validation
val_transform = A.Compose([
    A.Normalize(),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Define transformations for test
test_transform = A.Compose([
    A.Normalize(),
    ToTensorV2()
])

# Create datasets
train_dirs = [TRAIN_IMG_DIR]
dataset = CellSegmentationDataset(train_dirs, transform=train_transform)

# Split dataset to train and val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Change validation dataset transform
val_dataset.dataset.transform = val_transform

# Define test dataset
test_dataset = CellSegmentationDataset([TEST_IMG_DIR], transform=test_transform, train=False)

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)), num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

def get_model(num_classes):
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights)
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    
    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    
    # Modify anchor sizes as appropriate for cell detection
    model.rpn.anchor_generator.sizes = ((8,), (16,), (32,), (64,), (128,))
    
    # Increase proposals for better recall
    model.rpn.pre_nms_top_n_train = 2000
    model.rpn.post_nms_top_n_train = 1000
    model.rpn.pre_nms_top_n_test = 1000
    model.rpn.post_nms_top_n_test = 500
    
    # Adjust NMS threshold for better overlapping mask handling
    model.rpn.nms_thresh = 0.7
    
    # Increase detections per image
    model.roi_heads.detections_per_img = 300
    
    # Print model parameters for verification
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    return model

# Initialize the model
num_classes = len(CLASS_NAMES)
model = get_model(num_classes)
model.to(device)

# Define optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=0.00001
)

# Training function
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    
    for images, targets, _ in tqdm(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
    
    return total_loss / len(data_loader)

# Validation function
def validate(model, data_loader, device):
    training = model.training
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, targets, _ in tqdm(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Set model to training mode temporarily to compute losses
            model.train()
            loss_dict = model(images, targets)
            model.eval()  # Set back to eval mode
            
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
    
    # Restore the model's original state
    model.train(training)
    
    return total_loss / len(data_loader)

# Add early stopping to prevent overfitting
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
early_stopping = EarlyStopping(patience=5)

# Train the model
num_epochs = 30
best_val_loss = float('inf')

train_losses = []
val_losses = []

# Create a learning rate finder to find optimal learning rate
lr_find_epoch = 5

for epoch in range(num_epochs):
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{num_epochs} (LR: {current_lr:.6f})")
    
    # Train
    train_loss = train_one_epoch(model, optimizer, train_loader, device)
    train_losses.append(train_loss)
    
    # Validate
    val_loss = validate(model, val_loader, device)
    val_losses.append(val_loss)
    
    # Update learning rate
    if epoch >= lr_find_epoch:
        lr_scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
        print(f"New best model saved with validation loss: {val_loss:.4f}")
    
    # Check early stopping
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR,'loss_plot.png'))
plt.show()

model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth')))
model.eval()

# Define test-time augmentation (TTA) transforms
tta_transforms = [
    A.Compose([A.Normalize(), ToTensorV2()]),  # Original image
    A.Compose([A.HorizontalFlip(p=1.0), A.Normalize(), ToTensorV2()]),  # Horizontal flip
    A.Compose([A.VerticalFlip(p=1.0), A.Normalize(), ToTensorV2()]),  # Vertical flip
    A.Compose([A.Rotate(limit=90, p=1.0), A.Normalize(), ToTensorV2()]),  # 90-degree rotation
]

def apply_tta(model, image_tensor, device):
    # Store predictions
    all_boxes = []
    all_scores = []
    all_labels = []
    all_masks = []
    
    # Original image prediction
    with torch.no_grad():
        prediction = model([image_tensor.to(device)])[0]
        
    # Get original predictions
    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    masks = prediction['masks'].squeeze(1).cpu().numpy() > 0.5
    
    # Filter by confidence
    keep = scores > 0.3
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    masks = masks[keep]
    
    # Add original predictions
    if len(boxes) > 0:
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)
        all_masks.append(masks)
    
    # Try horizontal flip
    flipped_tensor = torch.flip(image_tensor, [2])
    
    with torch.no_grad():
        try:
            prediction = model([flipped_tensor.to(device)])[0]
            
            # Get flipped predictions
            boxes = prediction['boxes'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()
            masks = prediction['masks'].squeeze(1).cpu().numpy() > 0.5
            
            # Filter by confidence
            keep = scores > 0.3
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            masks = masks[keep]
            
            # Reverse the flip for boxes and masks
            if len(boxes) > 0:
                width = image_tensor.shape[3]
                
                # Flip boxes back
                flipped_boxes = boxes.copy()
                flipped_boxes[:, 0] = width - boxes[:, 2]
                flipped_boxes[:, 2] = width - boxes[:, 0]
                
                # Flip masks back
                flipped_masks = []
                for mask in masks:
                    flipped_mask = np.flip(mask, axis=1)
                    flipped_masks.append(flipped_mask)
                
                all_boxes.append(flipped_boxes)
                all_scores.append(scores)
                all_labels.append(labels)
                all_masks.append(np.array(flipped_masks))
        except Exception as e:
            print(f"Error in horizontal flip: {e}")
    
    # Combine predictions
    if not all_boxes:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    combined_boxes = np.vstack(all_boxes)
    combined_scores = np.concatenate(all_scores)
    combined_labels = np.concatenate(all_labels)
    combined_masks = np.vstack([m for masks in all_masks for m in masks]) if all_masks[0].shape[0] > 0 else np.array([])
    
    # Apply NMS (simplified version)
    final_indices = []
    
    # Group by class
    for class_id in np.unique(combined_labels):
        class_mask = combined_labels == class_id
        class_boxes = combined_boxes[class_mask]
        class_scores = combined_scores[class_mask]
        
        # Sort by score
        score_order = np.argsort(-class_scores)
        class_boxes = class_boxes[score_order]
        class_scores = class_scores[score_order]
        
        # Get indices in combined arrays
        class_indices = np.where(class_mask)[0][score_order]
        
        # Keep track of which boxes to keep
        keep = []
        
        while len(class_indices) > 0:
            keep.append(class_indices[0])
            
            if len(class_indices) == 1:
                break
            
            # Calculate IoU with the top box
            box = class_boxes[0]
            other_boxes = class_boxes[1:]
            
            # Calculate IoU
            x1 = np.maximum(box[0], other_boxes[:, 0])
            y1 = np.maximum(box[1], other_boxes[:, 1])
            x2 = np.minimum(box[2], other_boxes[:, 2])
            y2 = np.minimum(box[3], other_boxes[:, 3])
            
            w = np.maximum(0, x2 - x1)
            h = np.maximum(0, y2 - y1)
            
            overlap = w * h
            
            area1 = (box[2] - box[0]) * (box[3] - box[1])
            area2 = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
            
            iou = overlap / (area1 + area2 - overlap)
            
            # Remove boxes with IoU above threshold
            inds = np.where(iou <= 0.5)[0]
            
            class_indices = class_indices[1:][inds]
            class_boxes = other_boxes[inds]
        
        final_indices.extend(keep)
    
    # Get final predictions
    final_boxes = combined_boxes[final_indices]
    final_scores = combined_scores[final_indices]
    final_labels = combined_labels[final_indices]
    final_masks = combined_masks[final_indices] if len(combined_masks) > 0 else np.array([])
    
    return final_boxes, final_scores, final_labels, final_masks

# After training, find optimal thresholds for each class
def find_optimal_thresholds(model, val_loader, device):
    class_predictions = {1: [], 2: [], 3: [], 4: []}
    class_targets = {1: [], 2: [], 3: [], 4: []}
    
    # Get predictions for validation set
    model.eval()
    with torch.no_grad():
        for images, targets, _ in tqdm(val_loader):
            images = list(image.to(device) for image in images)
            outputs = model(images)
            
            # Process each image
            for i, output in enumerate(outputs):
                target = targets[i]
                
                # Get ground truth by class
                target_labels = target['labels'].cpu().numpy()
                target_masks = target['masks'].cpu().numpy()
                
                for class_id in range(1, 5):
                    class_indices = target_labels == class_id
                    if np.any(class_indices):
                        class_targets[class_id].append({
                            'masks': target_masks[class_indices]
                        })
                
                # Get predictions
                pred_labels = output['labels'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                pred_masks = output['masks'].squeeze(1).cpu().numpy() > 0.5
                
                for class_id in range(1, 5):
                    class_indices = pred_labels == class_id
                    if np.any(class_indices):
                        class_predictions[class_id].append({
                            'scores': pred_scores[class_indices],
                            'masks': pred_masks[class_indices]
                        })
    
    # Find optimal threshold for each class
    optimal_thresholds = {}
    for class_id in range(1, 5):
        best_ap = 0
        best_threshold = 0.5
        
        for threshold in [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
            # Calculate AP for this class with this threshold
            ap = calculate_class_ap(
                class_predictions[class_id], 
                class_targets[class_id], 
                threshold,
                iou_threshold=0.5
            )
            
            if ap > best_ap:
                best_ap = ap
                best_threshold = threshold
        
        optimal_thresholds[class_id] = best_threshold
        print(f"Class {class_id}: Optimal threshold = {best_threshold}, AP = {best_ap:.4f}")
    
    return optimal_thresholds

def adjust_score_by_size(score, box, image_width, image_height):
    width = box[2] - box[0]
    height = box[3] - box[1]
    box_area = width * height
    
    # Calculate image area
    image_area = image_width * image_height
    
    # Calculate box relative size
    relative_size = box_area / image_area
    
    # Boost score for small objects
    if relative_size < 0.01:
        boost_factor = 1.15
    elif relative_size < 0.05:
        boost_factor = 1.10
    else:  # No boost for normal objects
        boost_factor = 1.0
    
    # Apply boost
    adjusted_score = min(score * boost_factor, 1.0)
    
    return adjusted_score

def enhanced_mask_refinement(mask, box=None):
    mask_uint8 = mask.astype(np.uint8)
    h, w = mask_uint8.shape
    
    # Apply different processing based on mask size
    if h * w > 1000:
        kernel = np.ones((5, 5), np.uint8)
        mask_refined = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask_refined = cv2.morphologyEx(mask_refined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    else:  # Smaller masks
        kernel = np.ones((3, 3), np.uint8)
        mask_refined = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # If box is provided, ensure mask is contained within box
    if box is not None:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        box_mask = np.zeros_like(mask_uint8)
        box_mask[y1:y2, x1:x2] = 1
        mask_refined = mask_refined * box_mask
    
    return mask_refined > 0

def refine_mask_boundaries(mask, image):
    # Convert image to grayscale if it's color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Calculate image gradient
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Find edges in mask
    mask_uint8 = mask.astype(np.uint8)
    mask_edges = cv2.Canny(mask_uint8, 100, 200)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Create refined mask
    refined_mask = mask_uint8.copy()
    
    # Refine each contour
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        contour_mask = np.zeros_like(mask_uint8)
        cv2.drawContours(contour_mask, [largest_contour], -1, 1, -1)
        kernel = np.ones((3, 3), np.uint8)
        contour_mask = cv2.morphologyEx(contour_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Use the refined contour mask
        refined_mask = contour_mask
    
    return refined_mask > 0

# Generate predictions with all improvements
results = []

# Set thresholds for each class
class_thresholds = {
    1: 0.25,
    2: 0.22,
    3: 0.20,
    4: 0.18
}

with torch.no_grad():
    for images, img_paths in tqdm(test_loader):
        image_tensor = images[0].to(device)
        img_path = img_paths[0]
        img_name = Path(img_path).name
        
        image_id = None
        for item in test_image_to_id:
            if item['file_name'] == img_name:
                image_id = item['id']
                img_height = item['height']
                img_width = item['width']
                break
        
        if image_id is None:
            print(f"Warning: Could not find image_id for {img_name}")
            continue
        
        try:
            orig_image = image_tensor.cpu().permute(1, 2, 0).numpy()
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            orig_image = std * orig_image + mean
            orig_image = np.clip(orig_image * 255, 0, 255).astype(np.uint8)
            
            # Direct prediction
            prediction = model([image_tensor])[0]
            
            # Get predictions
            boxes = prediction['boxes'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()
            masks = prediction['masks'].squeeze(1).cpu().numpy() > 0.5
            
            # Process predictions
            for i in range(len(boxes)):
                box = boxes[i]
                score = scores[i]
                label = labels[i]
                
                # Apply class-specific threshold
                threshold = class_thresholds.get(int(label), 0.25)
                
                # Adjust score by object size
                adjusted_score = adjust_score_by_size(score, box, img_width, img_height)
                
                if adjusted_score < threshold:
                    continue
                
                # Get mask
                mask = masks[i]
                
                # Apply enhanced mask refinement
                mask_refined = enhanced_mask_refinement(mask, box)
                
                # Try boundary refinement if the mask is not too large
                if np.sum(mask_refined) < 5000:
                    try:
                        # Resize mask to image size if needed
                        if mask_refined.shape[0] != img_height or mask_refined.shape[1] != img_width:
                            mask_temp = cv2.resize(mask_refined.astype(np.uint8), (img_width, img_height), 
                                           interpolation=cv2.INTER_NEAREST)
                        else:
                            mask_temp = mask_refined
                        
                        # Apply boundary refinement
                        mask_final = refine_mask_boundaries(mask_temp, orig_image)

                    except Exception as e:
                        print(f"Error in boundary refinement: {e}")
                        mask_final = mask_refined

                else:
                    mask_final = mask_refined
                
                # Ensure mask is properly sized
                if mask_final.shape[0] != img_height or mask_final.shape[1] != img_width:
                    mask_final = cv2.resize(mask_final.astype(np.uint8), (img_width, img_height), 
                                    interpolation=cv2.INTER_NEAREST)
                
                bbox = [
                    float(box[0]), 
                    float(box[1]), 
                    float(box[2]-box[0]), 
                    float(box[3]-box[1])
                ]
                
                try:
                    rle_mask = encode_mask(mask_final > 0)
                    results.append({
                        'image_id': int(image_id),
                        'bbox': bbox,
                        'score': float(adjusted_score),
                        'category_id': int(label),
                        'segmentation': rle_mask
                    })
                    
                except Exception as e:
                    print(f"Error encoding mask: {e}")
        
        except Exception as e:
            print(f"Error processing image {img_name}: {e}")
            continue

# Save results to JSON file
with open(os.path.join(OUTPUT_DIR, 'test-results.json'), 'w') as f:
    json.dump(results, f)

print(f"Predictions saved to {OUTPUT_DIR}")
print(f"Total number of predictions: {len(results)}")

# Count predictions by class
class_counts = {}
for result in results:
    class_id = result['category_id']
    if class_id not in class_counts:
        class_counts[class_id] = 0
    class_counts[class_id] += 1

print("Predictions by class:")
for class_id, count in sorted(class_counts.items()):
    print(f"Class {class_id}: {count} predictions")

# Display a few test predictions for visual verification
def visualize_predictions(model, test_loader, num_samples=3):
    model.eval()
    fig, axs = plt.subplots(num_samples, 2, figsize=(12, num_samples*4))
    
    with torch.no_grad():
        for i, (images, img_paths) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            # Get image and run prediction
            image = images[0].to(device)
            predictions = model([image])
            
            # Get original image for visualization
            orig_image = images[0].permute(1, 2, 0).cpu().numpy()
            
            # Denormalize image
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            orig_image = std * orig_image + mean
            orig_image = np.clip(orig_image, 0, 1)
            
            # Display original image
            axs[i, 0].imshow(orig_image)
            axs[i, 0].set_title('Original Image')
            axs[i, 0].axis('off')
            
            # Display predictions
            pred_image = orig_image.copy()
            pred = predictions[0]
            
            # Draw masks and boxes
            for j in range(len(pred['boxes'])):
                score = pred['scores'][j].item()
                if score > 0.4:
                    mask = pred['masks'][j, 0].cpu().numpy() > 0.5
                    label = pred['labels'][j].item()
                    box = pred['boxes'][j].cpu().numpy().astype(np.int32)
                    
                    # Apply colored mask
                    color = np.array([random.random(), random.random(), random.random()])
                    pred_image = np.where(
                        np.repeat(mask[:, :, np.newaxis], 3, axis=2),
                        pred_image * 0.5 + color * 0.5,
                        pred_image
                    )
                    
                    # Draw bounding box
                    cv2.rectangle(pred_image, (box[0], box[1]), (box[2], box[3]), color, 2)
                    
                    # Add label
                    cv2.putText(pred_image, f"{CLASS_NAMES[label]}: {score:.2f}", 
                                (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, color, 2)
            
            # Show prediction
            axs[i, 1].imshow(pred_image)
            axs[i, 1].set_title('Predictions')
            axs[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Visualize a few predictions
visualize_predictions(model, test_loader)

# Zip submission file
results_file_path = os.path.join(OUTPUT_DIR, 'test-results.json')
if os.path.exists(results_file_path):
    submission_file_path = os.path.join(OUTPUT_DIR, 'submission.zip')
    with open(results_file_path, 'r') as f:
        predictions = json.load(f)
    
    # Print number of predictions and distribution of classes
    print(f"Total predictions: {len(predictions)}")
    class_counts = {}
    for pred in predictions:
        class_id = pred['category_id']
        if class_id not in class_counts:
            class_counts[class_id] = 0
        class_counts[class_id] += 1
    
    for class_id, count in sorted(class_counts.items()):
        print(f"Class {class_id}: {count} predictions")
    
    with zipfile.ZipFile(submission_file_path, 'w') as zipf:
        zipf.write(results_file_path, arcname='test-results.json')
    
    print(f"Submission file created at {submission_file_path}")
else:
    print(f"Results file not found at {results_file_path}")