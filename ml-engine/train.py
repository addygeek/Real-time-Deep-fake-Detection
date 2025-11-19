"""
Generate synthetic dataset and train the lightweight model
This creates a small dataset for demonstration purposes
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
import random
from tqdm import tqdm

from detector import SimpleCNN

class SyntheticFaceDataset(Dataset):
    """Generate synthetic face images for training"""
    
    def __init__(self, num_samples=1000, img_size=128):
        self.num_samples = num_samples
        self.img_size = img_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate synthetic face-like image
        img = self._generate_face(idx)
        
        # Label: 0 = real, 1 = fake
        # Use deterministic labeling based on index
        label = 1.0 if idx % 2 == 0 else 0.0
        
        # Add noise to fake images
        if label == 1.0:
            img = self._add_artifacts(img)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        
        return torch.from_numpy(img), torch.tensor([label], dtype=torch.float32)
    
    def _generate_face(self, seed):
        """Generate a synthetic face-like image"""
        np.random.seed(seed)
        
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Skin tone background
        skin_color = (random.randint(180, 220), random.randint(150, 200), random.randint(120, 180))
        img[:] = skin_color
        
        # Face oval
        center = (self.img_size // 2, self.img_size // 2)
        axes = (self.img_size // 3, self.img_size // 2)
        cv2.ellipse(img, center, axes, 0, 0, 360, skin_color, -1)
        
        # Eyes
        eye_y = self.img_size // 3
        eye_spacing = self.img_size // 4
        cv2.circle(img, (center[0] - eye_spacing, eye_y), 8, (50, 50, 50), -1)
        cv2.circle(img, (center[0] + eye_spacing, eye_y), 8, (50, 50, 50), -1)
        
        # Nose
        nose_pts = np.array([
            [center[0], center[1] - 10],
            [center[0] - 5, center[1] + 10],
            [center[0] + 5, center[1] + 10]
        ], np.int32)
        cv2.fillPoly(img, [nose_pts], (160, 130, 110))
        
        # Mouth
        mouth_y = int(self.img_size * 0.7)
        cv2.ellipse(img, (center[0], mouth_y), (20, 10), 0, 0, 180, (100, 50, 50), 2)
        
        # Add some texture
        noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return img
    
    def _add_artifacts(self, img):
        """Add deepfake-like artifacts"""
        # Add blur
        if random.random() > 0.5:
            img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Add compression artifacts
        if random.random() > 0.5:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(30, 60)]
            _, encimg = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(encimg, 1)
        
        # Add color shift
        if random.random() > 0.5:
            shift = random.randint(-20, 20)
            img = np.clip(img.astype(np.int16) + shift, 0, 255).astype(np.uint8)
        
        return img


def train_model(num_epochs=10, batch_size=32, save_path='models/frame_classifier.pth'):
    """Train the lightweight model"""
    
    print("=" * 70)
    print("Training SpectraShield Lightweight Model")
    print("=" * 70)
    
    # Create dataset
    print("\n[1/5] Creating synthetic dataset...")
    train_dataset = SyntheticFaceDataset(num_samples=1000)
    val_dataset = SyntheticFaceDataset(num_samples=200)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    
    # Initialize model
    print("\n[2/5] Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")
    
    model = SimpleCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print(f"\n[3/5] Training for {num_epochs} epochs...")
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path(save_path).parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
    
    print(f"\n[4/5] Training complete!")
    print(f"✓ Best validation accuracy: {best_val_acc:.2f}%")
    print(f"✓ Model saved to: {save_path}")
    
    # Test the model
    print(f"\n[5/5] Testing model...")
    test_model(model, device)
    
    return model

def test_model(model, device):
    """Test the trained model"""
    model.eval()
    
    # Generate a few test samples
    test_dataset = SyntheticFaceDataset(num_samples=10)
    
    print("\nTest Results:")
    print("-" * 50)
    
    correct = 0
    with torch.no_grad():
        for i in range(len(test_dataset)):
            img, label = test_dataset[i]
            img = img.unsqueeze(0).to(device)
            
            output = model(img)
            pred = (torch.sigmoid(output) > 0.5).float().item()
            
            is_correct = (pred == label.item())
            correct += is_correct
            
            print(f"Sample {i+1}: True={int(label.item())}, Pred={int(pred)} {'✓' if is_correct else '✗'}")
    
    accuracy = 100 * correct / len(test_dataset)
    print("-" * 50)
    print(f"Test Accuracy: {accuracy:.1f}%")

if __name__ == "__main__":
    # Train the model
    model = train_model(num_epochs=10, batch_size=32)
    
    print("\n" + "=" * 70)
    print("✓ Training Complete!")
    print("=" * 70)
    print("\nYou can now use the trained model with:")
    print("  python api.py")
    print("\nOr test the detector with:")
    print("  python detector.py")
