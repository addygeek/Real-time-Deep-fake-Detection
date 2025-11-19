"""
Compression Resilient Embeddings - Complete Implementation
Two-stream CNN for RGB and noise residuals with denoising autoencoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResilientEmbedder(nn.Module):
    """
    Two-stream CNN for compression-resilient feature extraction
    Stream 1: RGB features
    Stream 2: Noise residual features
    """
    
    def __init__(self, embedding_dim=512):
        super(ResilientEmbedder, self).__init__()
        
        # RGB Stream - Using EfficientNet-like architecture
        self.rgb_stream = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            
            # Global pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, embedding_dim // 2)
        )
        
        # Noise Residual Stream
        self.noise_stream = nn.Sequential(
            # Block 1 - Sensitive to high-frequency artifacts
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Global pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, embedding_dim // 2)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def extract_noise_residual(self, x):
        """
        Extract high-frequency noise residual
        This captures compression artifacts
        """
        # Apply Gaussian blur to get low-frequency component
        kernel_size = 5
        sigma = 1.0
        
        # Create Gaussian kernel
        kernel = self._gaussian_kernel(kernel_size, sigma)
        kernel = kernel.to(x.device)
        
        # Apply convolution for each channel
        blurred = F.conv2d(x, kernel, padding=kernel_size//2, groups=3)
        
        # Residual = Original - Blurred (high-frequency component)
        residual = x - blurred
        
        return residual
    
    def _gaussian_kernel(self, kernel_size, sigma):
        """Create Gaussian kernel for blur"""
        x = torch.arange(kernel_size).float() - kernel_size // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        kernel = kernel / kernel.sum()
        
        # Expand for 3 channels
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(3, 1, 1, 1)
        
        return kernel
    
    def forward(self, x):
        """
        Args:
            x: Input image tensor [B, 3, H, W]
        Returns:
            embeddings: Compression-resilient features [B, embedding_dim]
        """
        # RGB features
        rgb_features = self.rgb_stream(x)
        
        # Extract and process noise residual
        noise_residual = self.extract_noise_residual(x)
        noise_features = self.noise_stream(noise_residual)
        
        # Concatenate both streams
        combined = torch.cat([rgb_features, noise_features], dim=1)
        
        # Fusion
        embeddings = self.fusion(combined)
        
        return embeddings


class DenoisingAutoencoder(nn.Module):
    """
    Denoising autoencoder for learning robust representations
    Trained to reconstruct clean images from compressed versions
    """
    
    def __init__(self, latent_dim=256):
        super(DenoisingAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # 224x224 -> 112x112
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 112x112 -> 56x56
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 56x56 -> 28x28
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 28x28 -> 14x14
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 14x14 -> 7x7
            nn.Conv2d(512, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 28x28 -> 56x56
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 56x56 -> 112x112
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 112x112 -> 224x224
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, x):
        """
        Args:
            x: Noisy/compressed input [B, 3, H, W]
        Returns:
            reconstructed: Denoised output [B, 3, H, W]
            latent: Latent representation [B, latent_dim, 7, 7]
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def encode(self, x):
        """Get latent representation only"""
        return self.encoder(x)


class CompressionResilientNetwork(nn.Module):
    """
    Complete network combining embedder and denoising autoencoder
    """
    
    def __init__(self, embedding_dim=512, num_classes=2):
        super(CompressionResilientNetwork, self).__init__()
        
        self.embedder = ResilientEmbedder(embedding_dim)
        self.denoiser = DenoisingAutoencoder(latent_dim=256)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim + 256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x, return_reconstruction=False):
        """
        Args:
            x: Input image [B, 3, H, W]
            return_reconstruction: Whether to return denoised image
        Returns:
            logits: Classification logits [B, num_classes]
            (optional) reconstructed: Denoised image
        """
        # Get compression-resilient embeddings
        embeddings = self.embedder(x)
        
        # Get denoised representation
        reconstructed, latent = self.denoiser(x)
        latent_flat = latent.flatten(1)
        
        # Combine features
        combined = torch.cat([embeddings, latent_flat], dim=1)
        
        # Classify
        logits = self.classifier(combined)
        
        if return_reconstruction:
            return logits, reconstructed
        return logits


# Utility function for feature extraction
def extract_resilient_features(model, images):
    """
    Extract compression-resilient features from images
    
    Args:
        model: Trained ResilientEmbedder or CompressionResilientNetwork
        images: Input images [B, 3, H, W]
    
    Returns:
        features: Extracted features [B, embedding_dim]
    """
    model.eval()
    with torch.no_grad():
        if isinstance(model, ResilientEmbedder):
            features = model(images)
        elif isinstance(model, CompressionResilientNetwork):
            features = model.embedder(images)
        else:
            raise ValueError("Model must be ResilientEmbedder or CompressionResilientNetwork")
    
    return features


if __name__ == "__main__":
    # Test the models
    print("Testing Compression Resilient Embeddings...")
    
    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    
    # Test ResilientEmbedder
    print("\n1. Testing ResilientEmbedder...")
    embedder = ResilientEmbedder(embedding_dim=512)
    embeddings = embedder(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {embeddings.shape}")
    print(f"   ✓ ResilientEmbedder working!")
    
    # Test DenoisingAutoencoder
    print("\n2. Testing DenoisingAutoencoder...")
    denoiser = DenoisingAutoencoder(latent_dim=256)
    reconstructed, latent = denoiser(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Reconstructed shape: {reconstructed.shape}")
    print(f"   Latent shape: {latent.shape}")
    print(f"   ✓ DenoisingAutoencoder working!")
    
    # Test Complete Network
    print("\n3. Testing CompressionResilientNetwork...")
    network = CompressionResilientNetwork(embedding_dim=512, num_classes=2)
    logits, recon = network(x, return_reconstruction=True)
    print(f"   Input shape: {x.shape}")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Reconstruction shape: {recon.shape}")
    print(f"   ✓ Complete network working!")
    
    # Count parameters
    total_params = sum(p.numel() for p in network.parameters())
    print(f"\n4. Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    
    print("\n✓ All tests passed!")
