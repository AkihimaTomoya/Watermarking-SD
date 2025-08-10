#!/usr/bin/env python3
"""
Steganography models for watermark embedding and extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class StegEncoder(nn.Module):
    """
    Encoder for embedding watermarks into images
    """
    def __init__(
        self,
        resolution: int = 512,
        image_channels: int = 3,
        fingerprint_size: int = 48,
        return_residual: bool = False,
    ):
        super(StegEncoder, self).__init__()
        self.fingerprint_size = fingerprint_size
        self.image_channels = image_channels
        self.return_residual = return_residual
        self.resolution = resolution
        
        # Calculate upsampling factor to reach target resolution from 64x64
        self.upsample_factor = resolution // 64
        
        self.secret_dense = nn.Linear(self.fingerprint_size, 64 * 64 * image_channels)
        self.fingerprint_upsample = nn.Upsample(scale_factor=self.upsample_factor, mode='bilinear')
        
        # Encoder network
        self.conv1 = nn.Conv2d(2 * image_channels, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 2, 1)
        
        # Decoder network
        self.pad6 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up6 = nn.Conv2d(256, 128, 2, 1)
        self.upsample6 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv6 = nn.Conv2d(128 + 128, 128, 3, 1, 1)
        
        self.pad7 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up7 = nn.Conv2d(128, 64, 2, 1)
        self.upsample7 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv7 = nn.Conv2d(64 + 64, 64, 3, 1, 1)
        
        self.pad8 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up8 = nn.Conv2d(64, 32, 2, 1)
        self.upsample8 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv8 = nn.Conv2d(32 + 32, 32, 3, 1, 1)
        
        self.pad9 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up9 = nn.Conv2d(32, 32, 2, 1)
        self.upsample9 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv9 = nn.Conv2d(32 + 32 + 2 * image_channels, 32, 3, 1, 1)
        
        self.conv10 = nn.Conv2d(32, 32, 3, 1, 1)
        self.residual = nn.Conv2d(32, image_channels, 1)

    def forward(self, fingerprint: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to embed watermark into image
        
        Args:
            fingerprint: Watermark tensor of shape (batch_size, fingerprint_size)
            image: Input image tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Watermarked image tensor
        """
        # Prepare fingerprint
        fingerprint = F.relu(self.secret_dense(fingerprint))
        fingerprint = fingerprint.view(-1, self.image_channels, 64, 64)
        fingerprint_enlarged = self.fingerprint_upsample(fingerprint)
        
        # Concatenate fingerprint and image
        inputs = torch.cat([fingerprint_enlarged, image], dim=1)
        
        # Encoder
        conv1 = F.relu(self.conv1(inputs))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv4 = F.relu(self.conv4(conv3))
        conv5 = F.relu(self.conv5(conv4))
        
        # Decoder with skip connections
        up6 = F.relu(self.up6(self.pad6(self.upsample6(conv5))))
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = F.relu(self.conv6(merge6))
        
        up7 = F.relu(self.up7(self.pad7(self.upsample7(conv6))))
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = F.relu(self.conv7(merge7))
        
        up8 = F.relu(self.up8(self.pad8(self.upsample8(conv7))))
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = F.relu(self.conv8(merge8))
        
        up9 = F.relu(self.up9(self.pad9(self.upsample9(conv8))))
        merge9 = torch.cat([conv1, up9, inputs], dim=1)
        conv9 = F.relu(self.conv9(merge9))
        
        conv10 = F.relu(self.conv10(conv9))
        residual = self.residual(conv10)
        
        if not self.return_residual:
            residual = torch.sigmoid(residual)
        
        return residual

    def embed_watermark(self, image: torch.Tensor, watermark: torch.Tensor) -> torch.Tensor:
        """
        Convenience method to embed watermark into image
        
        Args:
            image: Input image tensor
            watermark: Watermark tensor
        
        Returns:
            Watermarked image
        """
        return self.forward(watermark, image)


class StegDecoder(nn.Module):
    """
    Decoder for extracting watermarks from images
    """
    def __init__(
        self, 
        resolution: int = 512, 
        image_channels: int = 3, 
        fingerprint_size: int = 48
    ):
        super(StegDecoder, self).__init__()
        self.resolution = resolution
        self.image_channels = image_channels
        self.fingerprint_size = fingerprint_size
        
        self.decoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, 3, 2, 1),  # /2
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # /4
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1),  # /8
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # /16
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 2, 1),  # /32
            nn.ReLU(),
        )
        
        # Calculate final feature size
        final_size = resolution // 32
        final_features = final_size * final_size * 128
        
        self.dense = nn.Sequential(
            nn.Linear(final_features, 512),
            nn.ReLU(),
            nn.Linear(512, fingerprint_size),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract watermark from image
        
        Args:
            image: Input image tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Extracted watermark tensor of shape (batch_size, fingerprint_size)
        """
        x = self.decoder(image)
        x = x.view(x.size(0), -1)
        return self.dense(x)

    def extract_watermark(self, image: torch.Tensor) -> torch.Tensor:
        """
        Convenience method to extract watermark from image
        
        Args:
            image: Input image tensor
        
        Returns:
            Extracted watermark tensor
        """
        return self.forward(image)

    def convert_to_fp16(self):
        """Convert model to fp16"""
        return self.half()
    
    def convert_to_fp32(self):
        """Convert model back to fp32"""
        return self.float()


class StegModel(nn.Module):
    """
    Complete steganography model with encoder and decoder
    """
    def __init__(
        self,
        resolution: int = 512,
        image_channels: int = 3,
        fingerprint_size: int = 48,
        return_residual: bool = False
    ):
        super(StegModel, self).__init__()
        self.encoder = StegEncoder(
            resolution=resolution,
            image_channels=image_channels,
            fingerprint_size=fingerprint_size,
            return_residual=return_residual
        )
        self.decoder = StegDecoder(
            resolution=resolution,
            image_channels=image_channels,
            fingerprint_size=fingerprint_size
        )
        
        self.resolution = resolution
        self.image_channels = image_channels
        self.fingerprint_size = fingerprint_size

    def forward(self, image: torch.Tensor, watermark: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: embed and extract watermark
        
        Args:
            image: Input image tensor
            watermark: Watermark tensor
        
        Returns:
            Tuple of (watermarked_image, extracted_watermark)
        """
        watermarked_image = self.encoder(watermark, image)
        extracted_watermark = self.decoder(watermarked_image)
        return watermarked_image, extracted_watermark

    def embed(self, image: torch.Tensor, watermark: torch.Tensor) -> torch.Tensor:
        """Embed watermark into image"""
        return self.encoder(watermark, image)

    def extract(self, image: torch.Tensor) -> torch.Tensor:
        """Extract watermark from image"""
        return self.decoder(image)

    def save_models(self, save_dir: str, prefix: str = ""):
        """
        Save encoder and decoder separately
        
        Args:
            save_dir: Directory to save models
            prefix: Prefix for model filenames
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        encoder_path = os.path.join(save_dir, f"{prefix}encoder.pt")
        decoder_path = os.path.join(save_dir, f"{prefix}decoder.pt")
        
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)
        
        # Save model configuration
        config = {
            'resolution': self.resolution,
            'image_channels': self.image_channels,
            'fingerprint_size': self.fingerprint_size
        }
        
        import json
        config_path = os.path.join(save_dir, f"{prefix}config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Models saved to {save_dir}")
        print(f"  Encoder: {encoder_path}")
        print(f"  Decoder: {decoder_path}")
        print(f"  Config: {config_path}")

    @classmethod
    def load_models(cls, save_dir: str, prefix: str = "", device: str = 'cpu'):
        """
        Load encoder and decoder from saved files
        
        Args:
            save_dir: Directory containing saved models
            prefix: Prefix for model filenames
            device: Device to load models on
        
        Returns:
            StegModel instance with loaded weights
        """
        import os
        import json
        
        config_path = os.path.join(save_dir, f"{prefix}config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        model = cls(**config)
        
        encoder_path = os.path.join(save_dir, f"{prefix}encoder.pt")
        decoder_path = os.path.join(save_dir, f"{prefix}decoder.pt")
        
        model.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        model.decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        
        model = model.to(device)
        
        print(f"Models loaded from {save_dir}")
        return model


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Calculate PSNR between two images"""
    mse = F.mse_loss(img1, img2)
    if mse < 1e-10:
        return torch.tensor(100.0)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def calculate_watermark_similarity(wm1: torch.Tensor, wm2: torch.Tensor) -> torch.Tensor:
    """Calculate cosine similarity between watermarks"""
    return F.cosine_similarity(wm1, wm2, dim=1).mean()


def generate_random_watermark(batch_size: int, fingerprint_size: int, device: str = 'cpu') -> torch.Tensor:
    """Generate random watermark"""
    return torch.randn(batch_size, fingerprint_size, device=device)


def generate_deterministic_watermark(
    seed: int, 
    batch_size: int, 
    fingerprint_size: int, 
    device: str = 'cpu'
) -> torch.Tensor:
    """Generate deterministic watermark from seed"""
    rng = np.random.RandomState(seed)
    watermark = rng.randn(batch_size, fingerprint_size).astype(np.float32)
    return torch.from_numpy(watermark).to(device)
