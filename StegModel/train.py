#!/usr/bin/env python3
"""
Training script for steganography watermark models
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
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
import random
import os

from .model import StegModel, calculate_psnr, calculate_watermark_similarity, generate_random_watermark

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging disabled")

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available, using MSE for perceptual loss")


class WatermarkDataset(Dataset):
    """
    Dataset for training watermark encoder/decoder
    """
    def __init__(
        self, 
        image_dir: str, 
        resolution: int = 512,
        fingerprint_size: int = 48,
        augment: bool = True,
        supported_formats: list = None
    ):
        self.image_dir = Path(image_dir)
        self.resolution = resolution
        self.fingerprint_size = fingerprint_size
        self.augment = augment
        
        if supported_formats is None:
            supported_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        
        # Get all image files
        self.image_files = []
        for ext in supported_formats:
            self.image_files.extend(list(self.image_dir.glob(f"**/{ext}")))
            self.image_files.extend(list(self.image_dir.glob(f"**/{ext.upper()}")))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"Found {len(self.image_files)} images in {image_dir}")
        
        # Define transforms
        base_transforms = [
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        
        if augment:
            augment_transforms = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.RandomRotation(degrees=5),
            ]
            # Insert augmentation transforms before ToTensor
            self.transform = transforms.Compose([
                transforms.Resize((resolution, resolution)),
                *augment_transforms,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose(base_transforms)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a random other image instead
            idx = random.randint(0, len(self.image_files) - 1)
            return self.__getitem__(idx)
        
        # Generate random watermark
        watermark = torch.randn(self.fingerprint_size)
        
        return image, watermark


class WatermarkTrainer:
    """
    Trainer for watermark encoder/decoder
    """
    def __init__(
        self,
        model: StegModel,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        lambda_image: float = 1.0,
        lambda_watermark: float = 1.0,
        use_wandb: bool = False,
        project_name: str = "watermark-training"
    ):
        self.device = device
        self.model = model.to(device)
        self.lambda_image = lambda_image
        self.lambda_watermark = lambda_watermark
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-6
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        
        # LPIPS for perceptual loss (optional)
        if LPIPS_AVAILABLE:
            self.lpips_model = lpips.LPIPS(net='vgg').to(device)
            self.use_lpips = True
        else:
            self.use_lpips = False
        
        # Wandb setup
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(project=project_name)
            wandb.watch(self.model)
    
    def calculate_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Calculate LPIPS between two images"""
        if self.use_lpips:
            return self.lpips_model(img1, img2).mean()
        else:
            return torch.tensor(0.0, device=img1.device)
    
    def train_step(self, images: torch.Tensor, watermarks: torch.Tensor) -> Dict[str, float]:
        """Single training step"""
        # Move to device
        images = images.to(self.device)
        watermarks = watermarks.to(self.device)
        
        # Forward pass
        watermarked_images, extracted_watermarks = self.model(images, watermarks)
        
        # Compute losses
        # 1. Image reconstruction loss
        if self.use_lpips:
            image_loss = self.calculate_lpips(watermarked_images, images)
        else:
            image_loss = self.mse_loss(watermarked_images, images)
        
        # 2. Watermark extraction loss
        watermark_loss = self.mse_loss(extracted_watermarks, watermarks)
        
        # 3. Total loss
        total_loss = self.lambda_image * image_loss + self.lambda_watermark * watermark_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Calculate metrics
        psnr = calculate_psnr(watermarked_images, images)
        cosine_similarity = calculate_watermark_similarity(extracted_watermarks, watermarks)
        
        return {
            'total_loss': total_loss.item(),
            'image_loss': image_loss.item(),
            'watermark_loss': watermark_loss.item(),
            'psnr': psnr.item(),
            'cosine_similarity': cosine_similarity.item()
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation step"""
        self.model.eval()
        
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
                # Move to device
                images = images.to(self.device)
                watermarks = watermarks.to(self.device)
                
                # Forward pass
                watermarked_images, extracted_watermarks = self.model(images, watermarks)
                
                # Calculate losses
                if self.use_lpips:
                    image_loss = self.calculate_lpips(watermarked_images, images)
                else:
                    image_loss = self.mse_loss(watermarked_images, images)
                
                watermark_loss = self.mse_loss(extracted_watermarks, watermarks)
                total_loss = self.lambda_image * image_loss + self.lambda_watermark * watermark_loss
                
                # Calculate metrics
                psnr = calculate_psnr(watermarked_images, images)
                cosine_similarity = calculate_watermark_similarity(extracted_watermarks, watermarks)
                
                # Accumulate metrics
                total_metrics['total_loss'] += total_loss.item()
                total_metrics['image_loss'] += image_loss.item()
                total_metrics['watermark_loss'] += watermark_loss.item()
                total_metrics['psnr'] += psnr.item()
                total_metrics['cosine_similarity'] += cosine_similarity.item()
                
                num_batches += 1
        
        # Average metrics
        for key in total_metrics:
            total_metrics[key] /= num_batches
        
        self.model.train()
        return total_metrics
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: Optional[DataLoader], 
        num_epochs: int,
        save_dir: str,
        save_interval: int = 10,
        eval_interval: int = 5
    ):
        """Full training loop"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        best_similarity = 0
        training_history = []
        
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
            val_metrics = {}
            if val_loader and (epoch + 1) % eval_interval == 0:
                val_metrics = self.validate(val_loader)
                val_similarity = val_metrics['cosine_similarity']
            else:
                val_similarity = epoch_metrics['cosine_similarity']
            
            # Update learning rate
            self.scheduler.step()
            
            # Store training history
            epoch_data = {
                'epoch': epoch + 1,
                'train': epoch_metrics,
                'val': val_metrics if val_metrics else None,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            training_history.append(epoch_data)
            
            # Logging
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train - Loss: {epoch_metrics['total_loss']:.4f}, "
                  f"PSNR: {epoch_metrics['psnr']:.2f}, "
                  f"Sim: {epoch_metrics['cosine_similarity']:.3f}")
            if val_metrics:
                print(f"  Val   - Loss: {val_metrics['total_loss']:.4f}, "
                      f"PSNR: {val_metrics['psnr']:.2f}, "
                      f"Sim: {val_metrics['cosine_similarity']:.3f}")
            
            if self.use_wandb:
                log_dict = {f"train/{k}": v for k, v in epoch_metrics.items()}
                if val_metrics:
                    log_dict.update({f"val/{k}": v for k, v in val_metrics.items()})
                log_dict['epoch'] = epoch + 1
                log_dict['learning_rate'] = self.optimizer.param_groups[0]['lr']
                wandb.log(log_dict)
            
            # Save model
            should_save = False
            suffix = ""
            
            if (epoch + 1) % save_interval == 0:
                should_save = True
                suffix = f"_epoch_{epoch+1}"
            
            if val_similarity > best_similarity:
                best_similarity = val_similarity
                should_save = True
                suffix = "_best"
            
            if should_save:
                self.model.save_models(save_dir, prefix=suffix)
                
                # Save training history
                with open(save_dir / f"training_history{suffix}.json", 'w') as f:
                    json.dump(training_history, f, indent=2)
        
        print(f"Training completed! Best similarity: {best_similarity:.4f}")
        
        # Save final training history
        with open(save_dir / "training_history_final.json", 'w') as f:
            json.dump(training_history, f, indent=2)
        
        return training_history

    def plot_training_curves(self, history: list, save_path: str = None):
        """Plot training curves"""
        epochs = [h['epoch'] for h in history]
        train_loss = [h['train']['total_loss'] for h in history]
        train_psnr = [h['train']['psnr'] for h in history]
        train_sim = [h['train']['cosine_similarity'] for h in history]
        
        val_loss = [h['val']['total_loss'] if h['val'] else None for h in history]
        val_psnr = [h['val']['psnr'] if h['val'] else None for h in history]
        val_sim = [h['val']['cosine_similarity'] if h['val'] else None for h in history]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(epochs, train_loss, 'b-', label='Train')
        if any(v is not None for v in val_loss):
            val_epochs = [e for e, v in zip(epochs, val_loss) if v is not None]
            val_loss_clean = [v for v in val_loss if v is not None]
            axes[0, 0].plot(val_epochs, val_loss_clean, 'r-', label='Val')
        axes[0, 0].set_title('Total Loss')
