"""
Training Script for Compression Resilient Embeddings
Trains the two-stream network with denoising autoencoder
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path

from model import ResilientEmbedder, DenoisingAutoencoder, CompressionResilientNetwork
from augmentation import CompressionSimulator, PreprocessingPipeline, create_compression_pairs

class CompressionDataset(Dataset):
    """
    Dataset for training compression-resilient models
    Generates synthetic data with compression artifacts
    """
    
    def __init__(self, num_samples=1000, image_size=224, mode='train'):
        self.num_samples = num_samples
        self.image_size = image_size
        self.mode = mode
        self.compressor = CompressionSimulator()
        self.pipeline = PreprocessingPipeline(image_size, augment=(mode=='train'))
    
    def __len__(self):
        return self.num_samples
    
    def generate_synthetic_image(self, seed):
        """Generate a synthetic face-like image"""
        np.random.seed(seed)
        
        img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Random background color
        bg_color = np.random.randint(100, 200, size=3)
        img[:] = bg_color
        
        # Add some patterns
        for _ in range(5):
            x, y = np.random.randint(0, self.image_size, size=2)
            radius = np.random.randint(10, 50)
            color = np.random.randint(0, 255, size=3)
            import cv2
            cv2.circle(img, (x, y), radius, color.tolist(), -1)
        
        # Add texture
        noise = np.random.randint(-30, 30, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return img
    
    def __getitem__(self, idx):
        # Generate original image
        original = self.generate_synthetic_image(idx)
        
        # Create compressed version
        if self.mode == 'train':
            severity = np.random.choice(['light', 'medium', 'heavy'])
        else:
            severity = 'medium'
        
        compressed = self.compressor.simulate_social_media_compression(original, severity)
        
        # Preprocess
        original_tensor = self.pipeline.preprocess(original, apply_compression=False)
        compressed_tensor = self.pipeline.preprocess(compressed, apply_compression=False)
        
        # Label: 0 = real, 1 = fake
        # For this synthetic data, we'll use compression as proxy for manipulation
        label = 1 if severity in ['medium', 'heavy'] else 0
        
        return {
            'original': original_tensor,
            'compressed': compressed_tensor,
            'label': torch.tensor(label, dtype=torch.long)
        }


def train_denoising_autoencoder(model, train_loader, val_loader, num_epochs=20, device='cuda'):
    """
    Train the denoising autoencoder
    
    Args:
        model: DenoisingAutoencoder
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        device: 'cuda' or 'cpu'
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print("Training Denoising Autoencoder...")
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            compressed = batch['compressed'].to(device)
            original = batch['original'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed, _ = model(compressed)
            loss = criterion(reconstructed, original)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                compressed = batch['compressed'].to(device)
                original = batch['original'].to(device)
                
                reconstructed, _ = model(compressed)
                loss = criterion(reconstructed, original)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/denoising_autoencoder.pth')
            print(f"✓ Saved best model (Val Loss: {val_loss:.4f})")
    
    return model


def train_compression_resilient_network(model, train_loader, val_loader, num_epochs=30, device='cuda'):
    """
    Train the complete compression-resilient network
    
    Args:
        model: CompressionResilientNetwork
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        device: 'cuda' or 'cpu'
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    classification_criterion = nn.CrossEntropyLoss()
    reconstruction_criterion = nn.MSELoss()
    
    print("\nTraining Compression Resilient Network...")
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            compressed = batch['compressed'].to(device)
            original = batch['original'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, reconstructed = model(compressed, return_reconstruction=True)
            
            # Combined loss: classification + reconstruction
            cls_loss = classification_criterion(logits, labels)
            recon_loss = reconstruction_criterion(reconstructed, original)
            loss = cls_loss + 0.1 * recon_loss  # Weight reconstruction less
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                compressed = batch['compressed'].to(device)
                original = batch['original'].to(device)
                labels = batch['label'].to(device)
                
                logits, reconstructed = model(compressed, return_reconstruction=True)
                
                cls_loss = classification_criterion(logits, labels)
                recon_loss = reconstruction_criterion(reconstructed, original)
                loss = cls_loss + 0.1 * recon_loss
                
                val_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/compression_resilient_network.pth')
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
    
    return model


def main():
    """Main training function"""
    
    # Create models directory
    Path('models').mkdir(exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = CompressionDataset(num_samples=1000, mode='train')
    val_dataset = CompressionDataset(num_samples=200, mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Train denoising autoencoder first
    print("\n" + "="*60)
    print("STAGE 1: Training Denoising Autoencoder")
    print("="*60)
    
    denoiser = DenoisingAutoencoder(latent_dim=256)
    denoiser = train_denoising_autoencoder(
        denoiser, train_loader, val_loader, 
        num_epochs=10, device=device
    )
    
    # Train complete network
    print("\n" + "="*60)
    print("STAGE 2: Training Complete Network")
    print("="*60)
    
    network = CompressionResilientNetwork(embedding_dim=512, num_classes=2)
    
    # Load pre-trained denoiser
    network.denoiser.load_state_dict(denoiser.state_dict())
    print("✓ Loaded pre-trained denoiser")
    
    network = train_compression_resilient_network(
        network, train_loader, val_loader,
        num_epochs=20, device=device
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nSaved models:")
    print("  - models/denoising_autoencoder.pth")
    print("  - models/compression_resilient_network.pth")


if __name__ == "__main__":
    main()
