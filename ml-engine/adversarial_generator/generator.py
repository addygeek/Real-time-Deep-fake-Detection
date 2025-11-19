import torch
import torch.nn as nn

class DeepfakeGenerator(nn.Module):
    """
    Stub for a Generator model (e.g., StyleGAN-based or Diffusion-based).
    In a real scenario, this would be a complex pre-trained model.
    Here we implement a simple UNet-like structure for demonstration.
    """
    def __init__(self):
        super(DeepfakeGenerator, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Args:
            x: Real image tensor [Batch, 3, H, W]
        Returns:
            fake: Generated fake image [Batch, 3, H, W]
        """
        z = self.encoder(x)
        fake = self.decoder(z)
        return fake

if __name__ == "__main__":
    gen = DeepfakeGenerator()
    dummy = torch.randn(1, 3, 64, 64)
    out = gen(dummy)
    print(f"Generator output: {out.shape}")
