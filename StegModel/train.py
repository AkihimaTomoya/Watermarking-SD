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
import torchvision.utils as vutils
import numpy as np
from PIL import Image
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List
import random
import os

from .steg_model import StegModel, calculate_psnr, calculate_watermark_similarity, generate_random_watermark

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
    Enhanced trainer for watermark encoder/decoder with image sampling
    """
    def __init__(
        self,
        model: StegModel,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        lambda_image: float = 1.0,
        lambda_watermark: float = 1.0,
        use_wandb: bool = False,
        project_name: str = "watermark-training",
        save_samples: bool = True,
        num_samples: int = 8
    ):
        self.device = device
        self.model = model.to(device)
        self.lambda_image = lambda_image
        self.lambda_watermark = lambda_watermark
        self.save_samples = save_samples
        self.num_samples = num_samples
        
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
        
        # Fixed sample data for consistent visualization
        self.fixed_samples = None
        self.fixed_watermarks = None
        
        # Wandb setup
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(project=project_name)
            wandb.watch(self.model)
    
    def setup_fixed_samples(self, data_loader: DataLoader):
        """Setup fixed samples for consistent visualization across epochs"""
        print("Setting up fixed samples for visualization...")
        
        # Get a batch of samples
        sample_batch = next(iter(data_loader))
        images, watermarks = sample_batch
        
        # Take only the specified number of samples
        self.fixed_samples = images[:self.num_samples].to(self.device)
        self.fixed_watermarks = watermarks[:self.num_samples].to(self.device)
        
        print(f"Fixed {self.num_samples} samples for visualization")
    
    def denormalize_image(self, tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize image tensor from [-1,1] to [0,1]"""
        return (tensor + 1) / 2
    
    def save_sample_images(self, epoch: int, save_dir: Path, prefix: str = ""):
        """Save sample images showing original, watermarked, and difference"""
        if not self.save_samples or self.fixed_samples is None:
            return
        
        self.model.eval()
        
        with torch.no_grad():
            # Generate watermarked images and extract watermarks
            watermarked_images, extracted_watermarks = self.model(
                self.fixed_samples, self.fixed_watermarks
            )
            
            # Calculate difference (amplified for visualization)
            difference = torch.abs(watermarked_images - self.fixed_samples) * 10
            difference = torch.clamp(difference, 0, 1)
            
            # Denormalize images for saving
            original_imgs = self.denormalize_image(self.fixed_samples)
            watermarked_imgs = self.denormalize_image(watermarked_images)
            
            # Create comparison grid
            comparison_images = []
            for i in range(self.num_samples):
                comparison_images.extend([
                    original_imgs[i],
                    watermarked_imgs[i], 
                    difference[i]
                ])
            
            # Save grid
            grid = vutils.make_grid(
                comparison_images, 
                nrow=3,  # 3 images per row (original, watermarked, difference)
                padding=2,
                normalize=False,
                pad_value=1.0
            )
            
            # Create samples directory
            samples_dir = save_dir / "samples"
            samples_dir.mkdir(exist_ok=True)
            
            # Save image
            filename = f"{prefix}epoch_{epoch:04d}_samples.png" if prefix else f"epoch_{epoch:04d}_samples.png"
            vutils.save_image(grid, samples_dir / filename)
            
            # Calculate and save metrics for these samples
            psnr = calculate_psnr(watermarked_images, self.fixed_samples)
            cosine_sim = calculate_watermark_similarity(extracted_watermarks, self.fixed_watermarks)
            
            # Save detailed comparison for first sample
            if epoch % 20 == 0 or prefix == "best_":  # Save detailed view less frequently
                self.save_detailed_comparison(
                    epoch, samples_dir, 
                    original_imgs[0], watermarked_imgs[0], difference[0],
                    psnr.item(), cosine_sim.item(), prefix
                )
            
            # Log to wandb if available
            if self.use_wandb:
                wandb.log({
                    f"samples/comparison_grid": wandb.Image(grid),
                    f"samples/psnr": psnr.item(),
                    f"samples/cosine_similarity": cosine_sim.item()
                }, step=epoch)
        
        self.model.train()
    
    def save_detailed_comparison(self, epoch: int, save_dir: Path, 
                                original: torch.Tensor, watermarked: torch.Tensor, 
                                difference: torch.Tensor, psnr: float, 
                                cosine_sim: float, prefix: str = ""):
        """Save detailed comparison with metrics for a single image"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Convert tensors to numpy for matplotlib
        original_np = original.cpu().permute(1, 2, 0).numpy()
        watermarked_np = watermarked.cpu().permute(1, 2, 0).numpy()
        difference_np = difference.cpu().permute(1, 2, 0).numpy()
        
        # Plot images
        axes[0].imshow(original_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(watermarked_np)
        axes[1].set_title(f'Watermarked Image\nPSNR: {psnr:.2f}dB')
        axes[1].axis('off')
        
        axes[2].imshow(difference_np)
        axes[2].set_title(f'Difference (×10)\nSimilarity: {cosine_sim:.3f}')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        filename = f"{prefix}epoch_{epoch:04d}_detailed.png" if prefix else f"epoch_{epoch:04d}_detailed.png"
        plt.savefig(save_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
    
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
        """Full training loop with automatic image sampling"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Set up fixed samples for visualization
        if self.save_samples:
            self.setup_fixed_samples(train_loader)
        
        best_similarity = 0
        training_history = []
        
        # Save initial samples (epoch 0)
        if self.save_samples:
            self.save_sample_images(0, save_dir)
        
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
            
            # Save model and samples at save_interval
            should_save = False
            suffix = ""
            
            # Regular interval saves (both model and samples)
            if (epoch + 1) % save_interval == 0:
                should_save = True
                suffix = f"_epoch_{epoch+1}"
                # Save samples at same time as model
                if self.save_samples:
                    self.save_sample_images(epoch + 1, save_dir)
            
            # Best model saves (both model and samples)
            if val_similarity > best_similarity:
                best_similarity = val_similarity
                should_save = True
                suffix = "_best"
                # Save best samples
                if self.save_samples:
                    self.save_sample_images(epoch + 1, save_dir, "best_")
            
            # Save model
            if should_save:
                self.model.save_models(save_dir, prefix=suffix)
                
                # Save training history
                with open(save_dir / f"training_history{suffix}.json", 'w') as f:
                    json.dump(training_history, f, indent=2)
        
        print(f"Training completed! Best similarity: {best_similarity:.4f}")
        
        # Save final training history
        with open(save_dir / "training_history_final.json", 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Save final samples
        if self.save_samples:
            self.save_sample_images(num_epochs, save_dir, "final_")
        
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
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # PSNR
        axes[0, 1].plot(epochs, train_psnr, 'b-', label='Train')
        if any(v is not None for v in val_psnr):
            val_epochs = [e for e, v in zip(epochs, val_psnr) if v is not None]
            val_psnr_clean = [v for v in val_psnr if v is not None]
            axes[0, 1].plot(val_epochs, val_psnr_clean, 'r-', label='Val')
        axes[0, 1].set_title('PSNR')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('PSNR (dB)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Similarity
        axes[1, 0].plot(epochs, train_sim, 'b-', label='Train')
        if any(v is not None for v in val_sim):
            val_epochs = [e for e, v in zip(epochs, val_sim) if v is not None]
            val_sim_clean = [v for v in val_sim if v is not None]
            axes[1, 0].plot(val_epochs, val_sim_clean, 'r-', label='Val')
        axes[1, 0].set_title('Watermark Similarity')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Cosine Similarity')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        lrs = [h['lr'] for h in history]
        axes[1, 1].plot(epochs, lrs, 'g-')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.show()


def create_data_loaders(
    train_dir: str,
    val_dir: Optional[str] = None,
    batch_size: int = 8,
    resolution: int = 512,
    fingerprint_size: int = 48,
    num_workers: int = 4,
    train_split: float = 0.9
) -> tuple:
    """
    Create training and validation data loaders
    """
    # Training dataset
    train_dataset = WatermarkDataset(
        train_dir,
        resolution=resolution,
        fingerprint_size=fingerprint_size,
        augment=True
    )
    
    # Validation dataset
    if val_dir:
        val_dataset = WatermarkDataset(
            val_dir,
            resolution=resolution,
            fingerprint_size=fingerprint_size,
            augment=False
        )
    else:
        # Split training dataset
        train_size = int(train_split * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        # Create separate validation dataset without augmentation
        val_dataset.dataset = WatermarkDataset(
            train_dir,
            resolution=resolution,
            fingerprint_size=fingerprint_size,
            augment=False
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train watermark encoder/decoder with image sampling")
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
    parser.add_argument("--eval_interval", type=int, default=5, help="Evaluate on validation set every N epochs")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of sample images to save")
    parser.add_argument("--use_wandb", action='store_true', help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", default="watermark-training", help="W&B project name")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--train_split", type=float, default=0.9, help="Train/val split ratio (if no val_data_dir)")
    parser.add_argument("--resume", help="Path to checkpoint to resume training from")
    parser.add_argument("--plot_curves", action='store_true', help="Plot training curves at the end")
    
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        train_dir=args.data_dir,
        val_dir=args.val_data_dir,
        batch_size=args.batch_size,
        resolution=args.resolution,
        fingerprint_size=args.fingerprint_size,
        num_workers=args.num_workers,
        train_split=args.train_split
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    model = StegModel(
        resolution=args.resolution,
        image_channels=3,
        fingerprint_size=args.fingerprint_size
    )
    
    print(f"Encoder parameters: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"Decoder parameters: {sum(p.numel() for p in model.decoder.parameters()):,}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming training from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resumed from epoch {start_epoch}")
    
    # Create trainer
    trainer = WatermarkTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        lambda_image=args.lambda_image,
        lambda_watermark=args.lambda_watermark,
        use_wandb=args.use_wandb,
        project_name=args.wandb_project,
        save_samples=True,  # Always save samples
        num_samples=args.num_samples
    )
    
    # Start training
    print("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        save_dir=args.output_dir,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval
    )
    
    # Plot training curves
    if args.plot_curves:
        save_path = Path(args.output_dir) / "training_curves.png"
        trainer.plot_training_curves(history, save_path)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
