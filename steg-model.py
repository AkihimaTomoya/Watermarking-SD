#!/usr/bin/env python3
"""
Training script for watermark decoder that can work with Stable Diffusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import argparse
import json
from pathlib import Path
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any
import random
import os


class StegEncoder(nn.Module):
    """
    Encoder for embedding watermarks into images
    """
    def __init__(
        self,
        resolution=512,
        image_channels=3,
        fingerprint_size=48,
        return_residual=False,
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

    def forward(self, fingerprint, image):
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


class StegDecoder(nn.Module):
    """
    Decoder for extracting watermarks from images
    """
    def __init__(self, resolution=512, image_channels=3, fingerprint_size=48):
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

    def forward(self, image):
        x = self.decoder(image)
        x = x.view(x.size(0), -1)
        return self.dense(x)


class WatermarkDataset(Dataset):
    """
    Dataset for training watermark encoder/decoder
    """
    def __init__(
        self, 
        image_dir: str, 
        resolution: int = 512,
        fingerprint_size: int = 48,
        augment: bool = True
    ):
        self.image_dir = Path(image_dir)
        self.resolution = resolution
        self.fingerprint_size = fingerprint_size
        self.augment = augment
        
        # Get all image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_files.extend(list(self.image_dir.glob(f"**/{ext}")))
        
        print(f"Found {len(self.image_files)} images in {image_dir}")
        
        # Define transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((resolution, resolution)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Generate random watermark
        watermark = torch.randn(self.fingerprint_size)
        
        return image, watermark


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = F.mse_loss(img1, img2)
    if mse < 1e-10:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def calculate_lpips(img1, img2, lpips_model):
    """Calculate LPIPS between two images"""
    return lpips_model(img1, img2).mean()


class WatermarkTrainer:
    """
    Trainer for watermark encoder/decoder
    """
    def __init__(
        self,
        encoder: StegEncoder,
        decoder: StegDecoder,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        lambda_image: float = 1.0,
        lambda_watermark: float = 1.0,
        use_wandb: bool = False
    ):
        self.device = device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.lambda_image = lambda_image
        self.lambda_watermark = lambda_watermark
        
        # Optimizers
        self.encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
        
        # Schedulers
        self.encoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.encoder_optimizer, T_max=1000, eta_min=1e-6
        )
        self.decoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.decoder_optimizer, T_max=1000, eta_min=1e-6
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        
        # LPIPS for perceptual loss (optional)
        try:
            import lpips
            self.lpips_model = lpips.LPIPS(net='vgg').to(device)
            self.use_lpips = True
        except ImportError:
            print("LPIPS not available, using MSE for image reconstruction loss")
            self.use_lpips = False
        
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="watermark-training")
    
    def train_step(self, images, watermarks):
        """Single training step"""
        batch_size = images.size(0)
        
        # Move to device
        images = images.to(self.device)
        watermarks = watermarks.to(self.device)
        
        # Forward pass through encoder
        watermarked_images = self.encoder(watermarks, images)
        
        # Forward pass through decoder
        extracted_watermarks = self.decoder(watermarked_images)
        
        # Compute losses
        # 1. Image reconstruction loss
        if self.use_lpips:
            image_loss = calculate_lpips(watermarked_images, images, self.lpips_model)
        else:
            image_loss = self.mse_loss(watermarked_images, images)
        
        # 2. Watermark extraction loss
        watermark_loss = self.mse_loss(extracted_watermarks, watermarks)
        
        # 3. Total loss
        total_loss = self.lambda_image * image_loss + self.lambda_watermark * watermark_loss
        
        # Backward pass
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        
        # Calculate metrics
        psnr = calculate_psnr(watermarked_images, images)
        cosine_similarity = F.cosine_similarity(extracted_watermarks, watermarks, dim=1).mean()
        
        return {
            'total_loss': total_loss.item(),
            'image_loss': image_loss.item(),
            'watermark_loss': watermark_loss.item(),
            'psnr': psnr.item(),
            'cosine_similarity': cosine_similarity.item()
        }
    
    def validate(self, val_loader):
        """Validation step"""
        self.encoder.eval()
        self.decoder.eval()
        
        total_metrics = {
            'total_loss': 0,
            'image_loss': 0,
            'watermark_loss': 0,
            'psnr': 0,
            'cosine_similarity': 0
        }
        
        num_batches = 0
        
        with torch.no_grad():
            for images, watermarks in val_loader:
                metrics = self.train_step(images, watermarks)
                for key, value in metrics.items():
                    total_metrics[key] += value
                num_batches += 1
        
        # Average metrics
        for key in total_metrics:
            total_metrics[key] /= num_batches
        
        self.encoder.train()
        self.decoder.train()
        
        return total_metrics
    
    def train(
        self, 
        train_loader, 
        val_loader, 
        num_epochs: int,
        save_dir: str,
        save_interval: int = 10
    ):
        """Full training loop"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        best_similarity = 0
        
        for epoch in range(num_epochs):
            epoch_metrics = {
                'total_loss': 0,
                'image_loss': 0,
                'watermark_loss': 0,
                'psnr': 0,
                'cosine_similarity': 0
            }
            
            num_batches = 0
            
            # Training
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for images, watermarks in progress_bar:
                metrics = self.train_step(images, watermarks)
                
                for key, value in metrics.items():
                    epoch_metrics[key] += value
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f"{metrics['total_loss']:.4f}",
                    'PSNR': f"{metrics['psnr']:.2f}",
                    'Sim': f"{metrics['cosine_similarity']:.3f}"
                })
            
            # Average training metrics
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches
            
            # Validation
            if val_loader:
                val_metrics = self.validate(val_loader)
                val_similarity = val_metrics['cosine_similarity']
            else:
                val_metrics = {}
                val_similarity = epoch_metrics['cosine_similarity']
            
            # Update learning rates
            self.encoder_scheduler.step()
            self.decoder_scheduler.step()
            
            # Logging
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train - Loss: {epoch_metrics['total_loss']:.4f}, PSNR: {epoch_metrics['psnr']:.2f}, Sim: {epoch_metrics['cosine_similarity']:.3f}")
            if val_metrics:
                print(f"  Val   - Loss: {val_metrics['total_loss']:.4f}, PSNR: {val_metrics['psnr']:.2f}, Sim: {val_metrics['cosine_similarity']:.3f}")
            
            if self.use_wandb:
                log_dict = {f"train/{k}": v for k, v in epoch_metrics.items()}
                if val_metrics:
                    log_dict.update({f"val/{k}": v for k, v in val_metrics.items()})
                log_dict['epoch'] = epoch
                wandb.log(log_dict)
            
            # Save model
            if (epoch + 1) % save_interval == 0 or val_similarity > best_similarity:
                if val_similarity > best_similarity:
                    best_similarity = val_similarity
                    suffix = "_best"
                else:
                    suffix = f"_epoch_{epoch+1}"
                
                torch.save(self.encoder.state_dict(), save_dir / f"encoder{suffix}.pt")
                torch.save(self.decoder.state_dict(), save_dir / f"decoder{suffix}.pt")
                
                # Save training config
                config = {
                    'epoch': epoch + 1,
                    'resolution': self.encoder.resolution,
                    'fingerprint_size': self.encoder.fingerprint_size,
                    'image_channels': self.encoder.image_channels,
                    'lambda_image': self.lambda_image,
                    'lambda_watermark': self.lambda_watermark,
                    'best_similarity': best_similarity
                }
                
                with open(save_dir / f"config{suffix}.json", 'w') as f:
                    json.dump(config, f, indent=2)
        
        print(f"Training completed! Best similarity: {best_similarity:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train watermark encoder/decoder")
    parser.add_argument("--data_dir", required=True, help="Directory containing training images")
    parser.add_argument("--val_data_dir", help="Directory containing validation images")
    parser.add_argument("--output_dir", default="./watermark_models", help="Output directory for trained models")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--fingerprint_size", type=int, default=48, help="Watermark fingerprint size")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lambda_image", type=float, default=1.0, help="Weight for image reconstruction loss")
    parser.add_argument("--lambda_watermark", type=float, default=1.0, help="Weight for watermark extraction loss")
    parser.add_argument("--save_interval", type=int, default=10, help="Save model every N epochs")
    parser.add_argument("--use_wandb", action='store_true', help="Use Weights & Biases for logging")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = WatermarkDataset(
        args.data_dir, 
        resolution=args.resolution,
        fingerprint_size=args.fingerprint_size,
        augment=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        drop_last=True
    )
    
    # Validation dataset (optional)
    val_loader = None
    if args.val_data_dir:
        val_dataset = WatermarkDataset(
            args.val_data_dir,
            resolution=args.resolution,
            fingerprint_size=args.fingerprint_size,
            augment=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=True
        )
    
    # Create models
    encoder = StegEncoder(
        resolution=args.resolution,
        image_channels=3,
        fingerprint_size=args.fingerprint_size
    )
    
    decoder = StegDecoder(
        resolution=args.resolution,
        image_channels=3,
        fingerprint_size=args.fingerprint_size
    )
    
    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    
    # Create trainer
    trainer = WatermarkTrainer(
        encoder=encoder,
        decoder=decoder,
        device=device,
        learning_rate=args.learning_rate,
        lambda_image=args.lambda_image,
        lambda_watermark=args.lambda_watermark,
        use_wandb=args.use_wandb
    )
    
    # Start training
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        save_dir=args.output_dir,
        save_interval=args.save_interval
    )


if __name__ == "__main__":
    main()
