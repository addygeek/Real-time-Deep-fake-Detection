"""
Compression Simulation and Augmentation
Simulates social media compression artifacts for training
"""

import cv2
import numpy as np
import torch
from PIL import Image
import io
import random

class CompressionSimulator:
    """
    Simulates various compression artifacts found in social media videos
    """
    
    def __init__(self):
        self.jpeg_qualities = [30, 40, 50, 60, 70, 80, 90]
        self.blur_kernels = [(3, 3), (5, 5), (7, 7)]
        self.downscale_factors = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    def apply_jpeg_compression(self, image, quality=None):
        """
        Apply JPEG compression to simulate social media compression
        
        Args:
            image: numpy array [H, W, C] or PIL Image
            quality: JPEG quality (1-100), random if None
        
        Returns:
            compressed: Compressed image
        """
        if quality is None:
            quality = random.choice(self.jpeg_qualities)
        
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        
        # Compress
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer)
        
        # Convert back to numpy
        compressed = np.array(compressed)
        
        return compressed
    
    def apply_gaussian_blur(self, image, kernel_size=None):
        """
        Apply Gaussian blur to simulate motion blur or low quality
        
        Args:
            image: numpy array [H, W, C]
            kernel_size: tuple (h, w), random if None
        
        Returns:
            blurred: Blurred image
        """
        if kernel_size is None:
            kernel_size = random.choice(self.blur_kernels)
        
        blurred = cv2.GaussianBlur(image, kernel_size, 0)
        return blurred
    
    def apply_downscale_upscale(self, image, scale_factor=None):
        """
        Downscale then upscale to simulate resolution loss
        
        Args:
            image: numpy array [H, W, C]
            scale_factor: float (0-1), random if None
        
        Returns:
            rescaled: Rescaled image
        """
        if scale_factor is None:
            scale_factor = random.choice(self.downscale_factors)
        
        h, w = image.shape[:2]
        
        # Downscale
        small = cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)),
                          interpolation=cv2.INTER_AREA)
        
        # Upscale back
        rescaled = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return rescaled
    
    def apply_color_shift(self, image, shift_range=20):
        """
        Apply random color shift to simulate color compression
        
        Args:
            image: numpy array [H, W, C]
            shift_range: maximum shift value
        
        Returns:
            shifted: Color-shifted image
        """
        shift = np.random.randint(-shift_range, shift_range, size=3)
        shifted = np.clip(image.astype(np.int16) + shift, 0, 255).astype(np.uint8)
        return shifted
    
    def apply_noise(self, image, noise_level=10):
        """
        Add Gaussian noise to simulate sensor noise or compression artifacts
        
        Args:
            image: numpy array [H, W, C]
            noise_level: standard deviation of noise
        
        Returns:
            noisy: Noisy image
        """
        noise = np.random.normal(0, noise_level, image.shape)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy
    
    def apply_block_artifacts(self, image, block_size=8):
        """
        Simulate DCT block artifacts from JPEG/H.264 compression
        
        Args:
            image: numpy array [H, W, C]
            block_size: DCT block size (typically 8)
        
        Returns:
            blocked: Image with block artifacts
        """
        h, w = image.shape[:2]
        
        # Quantize in blocks
        blocked = image.copy()
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = image[i:i+block_size, j:j+block_size]
                # Simple quantization
                mean_color = block.mean(axis=(0, 1))
                blocked[i:i+block_size, j:j+block_size] = mean_color
        
        return blocked.astype(np.uint8)
    
    def simulate_social_media_compression(self, image, severity='medium'):
        """
        Apply multiple compression artifacts to simulate social media processing
        
        Args:
            image: numpy array [H, W, C] or PIL Image
            severity: 'light', 'medium', 'heavy'
        
        Returns:
            compressed: Heavily compressed image
        """
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        compressed = image.copy()
        
        if severity == 'light':
            # Light compression
            compressed = self.apply_jpeg_compression(compressed, quality=random.randint(70, 90))
            if random.random() > 0.5:
                compressed = self.apply_gaussian_blur(compressed, (3, 3))
        
        elif severity == 'medium':
            # Medium compression
            compressed = self.apply_downscale_upscale(compressed, scale_factor=random.uniform(0.7, 0.9))
            compressed = self.apply_jpeg_compression(compressed, quality=random.randint(50, 70))
            if random.random() > 0.5:
                compressed = self.apply_gaussian_blur(compressed, (5, 5))
            if random.random() > 0.7:
                compressed = self.apply_color_shift(compressed, shift_range=15)
        
        elif severity == 'heavy':
            # Heavy compression (worst case)
            compressed = self.apply_downscale_upscale(compressed, scale_factor=random.uniform(0.5, 0.7))
            compressed = self.apply_jpeg_compression(compressed, quality=random.randint(30, 50))
            compressed = self.apply_gaussian_blur(compressed, random.choice([(5, 5), (7, 7)]))
            compressed = self.apply_color_shift(compressed, shift_range=20)
            if random.random() > 0.5:
                compressed = self.apply_noise(compressed, noise_level=random.randint(5, 15))
        
        return compressed
    
    def augment_batch(self, images, p=0.5):
        """
        Apply random compression to a batch of images
        
        Args:
            images: torch tensor [B, C, H, W] or numpy array [B, H, W, C]
            p: probability of applying compression
        
        Returns:
            augmented: Augmented images in same format
        """
        is_torch = isinstance(images, torch.Tensor)
        
        if is_torch:
            # Convert to numpy
            images = images.permute(0, 2, 3, 1).cpu().numpy()
            images = (images * 255).astype(np.uint8)
        
        augmented = []
        for img in images:
            if random.random() < p:
                severity = random.choice(['light', 'medium', 'heavy'])
                img = self.simulate_social_media_compression(img, severity)
            augmented.append(img)
        
        augmented = np.stack(augmented)
        
        if is_torch:
            # Convert back to torch
            augmented = torch.from_numpy(augmented).float() / 255.0
            augmented = augmented.permute(0, 3, 1, 2)
        
        return augmented


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for compression-resilient training
    """
    
    def __init__(self, image_size=224, augment=True):
        self.image_size = image_size
        self.augment = augment
        self.compressor = CompressionSimulator()
    
    def preprocess(self, image, apply_compression=True):
        """
        Preprocess single image
        
        Args:
            image: PIL Image or numpy array
            apply_compression: Whether to apply compression simulation
        
        Returns:
            processed: Preprocessed tensor [C, H, W]
        """
        # Convert to numpy if PIL
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Apply compression simulation
        if apply_compression and self.augment:
            image = self.compressor.simulate_social_media_compression(
                image, 
                severity=random.choice(['light', 'medium', 'heavy'])
            )
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor [C, H, W]
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Normalize (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        
        return image
    
    def preprocess_batch(self, images, apply_compression=True):
        """
        Preprocess batch of images
        
        Args:
            images: List of PIL Images or numpy arrays
            apply_compression: Whether to apply compression simulation
        
        Returns:
            batch: Preprocessed tensor [B, C, H, W]
        """
        processed = [self.preprocess(img, apply_compression) for img in images]
        return torch.stack(processed)


def create_compression_pairs(image, num_pairs=5):
    """
    Create pairs of (original, compressed) images for training
    
    Args:
        image: Original image
        num_pairs: Number of compression variants to create
    
    Returns:
        pairs: List of (original, compressed) tuples
    """
    compressor = CompressionSimulator()
    pairs = []
    
    severities = ['light', 'medium', 'heavy']
    
    for i in range(num_pairs):
        severity = random.choice(severities)
        compressed = compressor.simulate_social_media_compression(image, severity)
        pairs.append((image, compressed))
    
    return pairs


if __name__ == "__main__":
    print("Testing Compression Simulation...")
    
    # Create test image
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Test CompressionSimulator
    print("\n1. Testing CompressionSimulator...")
    simulator = CompressionSimulator()
    
    # JPEG compression
    jpeg_compressed = simulator.apply_jpeg_compression(test_image, quality=50)
    print(f"   JPEG compression: {test_image.shape} -> {jpeg_compressed.shape}")
    
    # Blur
    blurred = simulator.apply_gaussian_blur(test_image)
    print(f"   Gaussian blur: {test_image.shape} -> {blurred.shape}")
    
    # Downscale-upscale
    rescaled = simulator.apply_downscale_upscale(test_image, scale_factor=0.5)
    print(f"   Downscale-upscale: {test_image.shape} -> {rescaled.shape}")
    
    # Social media simulation
    compressed = simulator.simulate_social_media_compression(test_image, 'heavy')
    print(f"   Social media compression: {test_image.shape} -> {compressed.shape}")
    print("   ✓ CompressionSimulator working!")
    
    # Test PreprocessingPipeline
    print("\n2. Testing PreprocessingPipeline...")
    pipeline = PreprocessingPipeline(image_size=224, augment=True)
    
    processed = pipeline.preprocess(test_image, apply_compression=True)
    print(f"   Preprocessed shape: {processed.shape}")
    print(f"   Value range: [{processed.min():.2f}, {processed.max():.2f}]")
    print("   ✓ PreprocessingPipeline working!")
    
    # Test batch processing
    print("\n3. Testing batch preprocessing...")
    batch_images = [test_image for _ in range(4)]
    batch = pipeline.preprocess_batch(batch_images, apply_compression=True)
    print(f"   Batch shape: {batch.shape}")
    print("   ✓ Batch preprocessing working!")
    
    # Test compression pairs
    print("\n4. Testing compression pairs...")
    pairs = create_compression_pairs(test_image, num_pairs=3)
    print(f"   Created {len(pairs)} pairs")
    print(f"   Pair 0: original {pairs[0][0].shape}, compressed {pairs[0][1].shape}")
    print("   ✓ Compression pairs working!")
    
    print("\n✓ All augmentation tests passed!")
