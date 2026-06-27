import os
import sys
import io

# Fix Windows encoding issue when PyTorch prints emojis (like ✅) during ONNX export
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import random
from tqdm import tqdm


# Setup logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("TrainExpressionModel")

# FERPlus index mapping:
# 0: neutral, 1: happiness, 2: surprise, 3: sadness, 4: anger, 5: disgust, 6: fear, 7: contempt
CLASS_MAPPING = {
    "neutral": 0,
    "happy": 1,
    "surprise": 2,
    "sad": 3,
    "angry": 4,
    "disgust": 5,
    "fear": 6
}

def preprocess_train(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, (64, 64))
    
    # 1. Random horizontal flip
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
        
    # 2. Random rotation (between -15 and 15 degrees)
    if random.random() > 0.5:
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((32, 32), angle, 1.0)
        img = cv2.warpAffine(img, M, (64, 64), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        
    img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
    return img_tensor

def preprocess_val(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, (64, 64))
    img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
    return img_tensor

class FERDataset(Dataset):
    def __init__(self, root_dir, max_samples_per_class=None, is_train=True):
        self.root_dir = root_dir
        self.is_train = is_train
        self.samples = []
        
        # Ensure reproducible sampling
        random.seed(42)
        
        for category in sorted(os.listdir(root_dir)):
            cat_dir = os.path.join(root_dir, category)
            if not os.path.isdir(cat_dir):
                continue
                
            label = CLASS_MAPPING.get(category.lower(), -1)
            if label == -1:
                continue
                
            cat_files = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            random.shuffle(cat_files)
            
            if max_samples_per_class is not None:
                cat_files = cat_files[:max_samples_per_class]
                
            for filename in cat_files:
                self.samples.append((os.path.join(cat_dir, filename), label))
                    
        logger.info(f"Loaded {len(self.samples)} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            if self.is_train:
                img_tensor = preprocess_train(img_path)
            else:
                img_tensor = preprocess_val(img_path)
                
            if img_tensor is None:
                img_tensor = torch.zeros((1, 64, 64), dtype=torch.float32)
        except Exception:
            img_tensor = torch.zeros((1, 64, 64), dtype=torch.float32)
            
        return img_tensor, label

class ExpressionCNN(nn.Module):
    def __init__(self):
        super(ExpressionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout(0.25)
        
        # 64x64 -> after four poolings -> 4x4
        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 8) # Output size 8 for FERPlus matching

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout_conv(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout_conv(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout_conv(x)
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout_conv(x)
        
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "data", "kaggle_dataset", "images", "images")
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "validation")
    output_model_path = os.path.join(base_dir, "data", "emotion-ferplus-8.onnx")
    
    # 2. Datasets & Dataloaders with subsetting (1200 per class train, 300 per class validation)
    logger.info("Initializing datasets...")
    train_dataset = FERDataset(train_dir, max_samples_per_class=1200, is_train=True)
    val_dataset = FERDataset(val_dir, max_samples_per_class=300, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
    
    # 3. Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = ExpressionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    epochs = 12
    logger.info(f"Starting training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        epoch_start = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} (Train)", unit="batch")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct_batch = predicted.eq(targets).sum().item()
            correct += correct_batch
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct_batch / targets.size(0) * 100:.2f}%"
            })
                
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct / total * 100
        
        # Validation evaluation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} (Val)  ", unit="batch", leave=False)
        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = val_correct / val_total * 100
        epoch_time = time.time() - epoch_start
        
        # Step the scheduler
        scheduler.step(epoch_val_loss)
        
        logger.info(f"Epoch {epoch}/{epochs} completed in {epoch_time:.1f}s | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        sys.stdout.flush()
        
    logger.info("Training finished. Exporting model to ONNX...")
    sys.stdout.flush()
    
    # 4. Export to ONNX format
    model.eval()
    dummy_input = torch.randn(1, 1, 64, 64).to(device)
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_model_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_1'],
            output_names=['output_1'],
            dynamic_axes={'input_1': {0: 'batch_size'}, 'output_1': {0: 'batch_size'}},
            dynamo=False
        )
        logger.info(f"Custom expression model successfully exported to ONNX format at: {output_model_path}")
        sys.stdout.flush()
    except Exception as e:
        logger.error(f"Failed to export model to ONNX: {e}")
        sys.stdout.flush()
        sys.exit(1)

if __name__ == "__main__":
    main()
