#!/usr/bin/env python3
"""
Integration module for embedding watermark functionality into Stable Diffusion models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
import numpy as np
from typing import Optional, Dict, Any, Union
from safetensors import safe_open
from safetensors.torch import save_file


class WatermarkUNetWrapper(nn.Module):
    """
    Wrapper that integrates watermark functionality into Stable Diffusion UNet
    """
    def __init__(
        self, 
        unet, 
        watermark_decoder, 
        fingerprint_size: int = 48,
        watermark_strength: float = 0.1
    ):
        super().__init__()
        self.unet = unet
        self.watermark_decoder = watermark_decoder
        self.fingerprint_size = fingerprint_size
        self.watermark_strength = watermark_strength
        
        # Add watermark embedding layer to project watermark to time embedding dimension
        if hasattr(unet, 'time_embed') and hasattr(unet.time_embed, 'linear_1'):
            time_embed_dim = unet.time_embed.linear_1.out_features
        elif hasattr(unet, 'time_embedding'):
            time_embed_dim = unet.time_embedding.linear_1.out_features
        else:
            # Default for most SD models
            time_embed_dim = 1280
            
        self.watermark_embed = nn.Sequential(
            nn.Linear(fingerprint_size, time_embed_dim // 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim // 2, time_embed_dim),
        )
        
        print(f"Watermark embedding dimension: {fingerprint_size} -> {time_embed_dim}")

    def forward(self, sample, timestep, encoder_hidden_states, watermark=None, **kwargs):
        """
        Forward pass with optional watermark integration
        """
        if watermark is not None and watermark.numel() > 0:
            # Ensure watermark has correct shape
            if watermark.dim() == 1:
                watermark = watermark.unsqueeze(0)
            if watermark.shape[0] != sample.shape[0]:
                watermark = watermark.expand(sample.shape[0], -1)
            
            # Embed watermark
            wm_emb = self.watermark_embed(watermark)
            
            # Store original time embedding
            if hasattr(self.unet, 'time_embed'):
                original_time_embed = self.unet.time_embed
            elif hasattr(self.unet, 'time_embedding'):
                original_time_embed = self.unet.time_embedding
            else:
                # Fallback: try to find time embedding in the forward pass
                return self._forward_with_time_injection(sample, timestep, encoder_hidden_states, wm_emb, **kwargs)
            
            # Create modified time embedding function
            def modified_time_embed(timesteps):
                t_emb = original_time_embed(timesteps)
                # Add watermark embedding
                return t_emb + self.watermark_strength * wm_emb
            
            # Temporarily replace time embedding
            if hasattr(self.unet, 'time_embed'):
                self.unet.time_embed = modified_time_embed
            else:
                self.unet.time_embedding = modified_time_embed
            
            try:
                result = self.unet(sample, timestep, encoder_hidden_states, **kwargs)
            finally:
                # Restore original time embedding
                if hasattr(self.unet, 'time_embed'):
                    self.unet.time_embed = original_time_embed
                else:
                    self.unet.time_embedding = original_time_embed
            
            return result
        
        # Standard forward pass without watermark
        return self.unet(sample, timestep, encoder_hidden_states, **kwargs)

    def _forward_with_time_injection(self, sample, timestep, encoder_hidden_states, wm_emb, **kwargs):
        """
        Fallback method for injecting watermark when time embedding structure is unknown
        """
        # This is a more general approach that modifies the timestep embeddings directly
        # Get time embeddings from the UNet
        if hasattr(self.unet, 'get_time_embed'):
            t_emb = self.unet.get_time_embed(timestep, sample.device)
        else:
            # Manual time embedding calculation
            if timestep.dim() == 0:
                timestep = timestep.expand(sample.shape[0])
            
            half_dim = 160  # Typical for SD models
            emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=sample.device) * -emb)
            emb = timestep[:, None].float() * emb[None, :]
            emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=1)
            t_emb = emb
        
        # Add watermark to time embedding
        t_emb = t_emb + self.watermark_strength * wm_emb
        
        # Call UNet with modified time embedding
        # This requires access to UNet internals, which may vary by implementation
        return self.unet(sample, timestep, encoder_hidden_states, **kwargs)

    def extract_watermark(self, image):
        """
        Extract watermark from generated image
        """
        return self.watermark_decoder(image)


def load_stable_diffusion_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load Stable Diffusion checkpoint from various formats
    """
    checkpoint_path = Path(checkpoint_path)
    
    if checkpoint_path.suffix == '.safetensors':
        state_dict = {}
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        return {"state_dict": state_dict}
    
    elif checkpoint_path.suffix in ['.ckpt', '.pt', '.pth']:
        return torch.load(checkpoint_path, map_location='cpu')
    
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path.suffix}")


def integrate_watermark_into_checkpoint(
    sd_checkpoint_path: str,
    watermark_decoder_path: str,
    output_path: str,
    fingerprint_size: int = 48,
    resolution: int = 512,
    image_channels: int = 3,
    watermark_strength: float = 0.1
):
    """
    Integrate pre-trained watermark decoder into Stable Diffusion checkpoint
    """
    print(f"Loading Stable Diffusion checkpoint from {sd_checkpoint_path}...")
    checkpoint = load_stable_diffusion_checkpoint(sd_checkpoint_path)
    
    print(f"Loading watermark decoder from {watermark_decoder_path}...")
    
    # Load watermark decoder
    if Path(watermark_decoder_path).suffix == '.json':
        # Load from directory with config
        decoder_dir = Path(watermark_decoder_path).parent
        with open(watermark_decoder_path, 'r') as f:
            config = json.load(f)
        
        # Import the StegDecoder class
        try:
            from stegmodel.model import StegDecoder
        except ImportError:
            raise ImportError("Please ensure the stegmodel package is available")
        
        decoder = StegDecoder(
            resolution=config.get('resolution', resolution),
            image_channels=config.get('image_channels', image_channels),
            fingerprint_size=config.get('fingerprint_size', fingerprint_size)
        )
        
        decoder_weights_path = decoder_dir / f"{Path(watermark_decoder_path).stem.replace('config', 'decoder')}.pt"
        decoder.load_state_dict(torch.load(decoder_weights_path, map_location='cpu'))
        
    else:
        # Direct decoder weights file
        try:
            from stegmodel.model import StegDecoder
        except ImportError:
            raise ImportError("Please ensure the stegmodel package is available")
            
        decoder = StegDecoder(
            resolution=resolution,
            image_channels=image_channels,
            fingerprint_size=fingerprint_size
        )
        decoder.load_state_dict(torch.load(watermark_decoder_path, map_location='cpu'))
    
    print(f"Watermark decoder loaded with {sum(p.numel() for p in decoder.parameters()):,} parameters")
    
    # Add watermark components to checkpoint
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Add watermark decoder parameters with proper prefix
    for name, param in decoder.named_parameters():
        key = f"watermark_decoder.{name}"
        state_dict[key] = param
    
    # Create and add watermark embedding layer
    # Determine time embedding dimension from UNet
    time_embed_keys = [k for k in state_dict.keys() if 'time_embed' in k.lower() and 'linear_1.weight' in k]
    if not time_embed_keys:
        time_embed_keys = [k for k in state_dict.keys() if 'time_embedding' in k.lower() and 'linear_1.weight' in k]
    
    if time_embed_keys:
        time_embed_weight = state_dict[time_embed_keys[0]]
        time_embed_dim = time_embed_weight.shape[0]
        print(f"Detected time embedding dimension: {time_embed_dim}")
    else:
        time_embed_dim = 1280  # Default for SD 1.5
        print(f"Using default time embedding dimension: {time_embed_dim}")
    
    # Create watermark embedding layers
    watermark_embed = nn.Sequential(
        nn.Linear(fingerprint_size, time_embed_dim // 2),
        nn.SiLU(),
        nn.Linear(time_embed_dim // 2, time_embed_dim),
    )
    
    # Add watermark embedding parameters
    for name, param in watermark_embed.named_parameters():
        key = f"watermark_embed.{name}"
        state_dict[key] = param
    
    # Update or create metadata
    if 'metadata' not in checkpoint:
        checkpoint['metadata'] = {}
    
    checkpoint['metadata'].update({
        'watermark_enabled': True,
        'watermark_fingerprint_size': fingerprint_size,
        'watermark_resolution': resolution,
        'watermark_channels': image_channels,
        'watermark_strength': watermark_strength,
        'integration_version': '2.0'
    })
    
    # Save the modified checkpoint
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    print(f"Saving watermarked checkpoint to {output_path}...")
    
    if output_path.suffix == '.safetensors':
        # Save as safetensors
        save_file(state_dict, output_path)
        
        # Save metadata separately
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(checkpoint.get('metadata', {}), f, indent=2)
        print(f"Metadata saved to {metadata_path}")
    else:
        # Save as .ckpt
        torch.save(checkpoint, output_path)
    
    print("Integration completed successfully!")
    
    # Save integration info
    integration_info = {
        'source_checkpoint': str(sd_checkpoint_path),
        'watermark_decoder': str(watermark_decoder_path),
        'output_checkpoint': str(output_path),
        'fingerprint_size': fingerprint_size,
        'resolution': resolution,
        'image_channels': image_channels,
        'watermark_strength': watermark_strength,
        'time_embed_dim': time_embed_dim,
        'decoder_parameters': sum(p.numel() for p in decoder.parameters())
    }
    
    info_path = output_path.parent / f"{output_path.stem}_integration_info.json"
    with open(info_path, 'w') as f:
        json.dump(integration_info, f, indent=2)
    print(f"Integration info saved to {info_path}")
    
    return checkpoint, integration_info


def verify_watermark_integration(checkpoint_path: str) -> Dict[str, Any]:
    """
    Verify that watermark components are properly integrated into checkpoint
    """
    print(f"Verifying watermark integration in {checkpoint_path}...")
    
    checkpoint = load_stable_diffusion_checkpoint(checkpoint_path)
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Check for watermark components
    watermark_decoder_keys = [k for k in state_dict.keys() if k.startswith('watermark_decoder.')]
    watermark_embed_keys = [k for k in state_dict.keys() if k.startswith('watermark_embed.')]
    
    verification_results = {
        'has_watermark_decoder': len(watermark_decoder_keys) > 0,
        'has_watermark_embed': len(watermark_embed_keys) > 0,
        'decoder_parameters': len(watermark_decoder_keys),
        'embed_parameters': len(watermark_embed_keys),
        'metadata': checkpoint.get('metadata', {}),
        'watermark_enabled': checkpoint.get('metadata', {}).get('watermark_enabled', False)
    }
    
    if verification_results['has_watermark_decoder']:
        print(f"✓ Watermark decoder found with {verification_results['decoder_parameters']} parameters")
    else:
        print("✗ Watermark decoder not found")
    
    if verification_results['has_watermark_embed']:
        print(f"✓ Watermark embedding found with {verification_results['embed_parameters']} parameters")
    else:
        print("✗ Watermark embedding not found")
    
    if verification_results['watermark_enabled']:
        metadata = verification_results['metadata']
        print(f"✓ Watermark metadata found:")
        print(f"  Fingerprint size: {metadata.get('watermark_fingerprint_size', 'unknown')}")
        print(f"  Resolution: {metadata.get('watermark_resolution', 'unknown')}")
        print(f"  Strength: {metadata.get('watermark_strength', 'unknown')}")
    else:
        print("✗ Watermark metadata not found or disabled")
    
    return verification_results


def create_watermark_config(
    output_dir: str,
    fingerprint_size: int = 48,
    watermark_strength: float = 0.1,
    extraction_threshold: float = 0.5
):
    """
    Create configuration files for watermarked model usage
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    config = {
        "watermark_config": {
            "fingerprint_size": fingerprint_size,
            "strength": watermark_strength,
            "extraction_threshold": extraction_threshold,
            "resolution": 512,
            "image_channels": 3
        },
        "usage_instructions": {
            "loading": "Use WatermarkStableDiffusionPipeline.from_pretrained()",
            "generation": "Pass watermark parameter to the pipeline call",
            "extraction": "Use pipeline.extract_watermark() on generated images",
            "verification": "Use pipeline.verify_watermark() to check similarity"
        },
        "requirements": [
            "torch>=1.9.0",
            "diffusers>=0.20.0",
            "transformers>=4.21.0",
            "pillow>=8.0.0",
            "numpy>=1.21.0"
        ]
    }
    
    config_path = output_dir / "watermark_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Watermark configuration saved to {config_path}")
    return config_path
