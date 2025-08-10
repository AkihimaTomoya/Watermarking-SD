#!/usr/bin/env python3
"""
Watermark-enabled Stable Diffusion Pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Dict, Any, Tuple
import numpy as np
from PIL import Image
import json
from pathlib import Path

try:
    from diffusers import (
        StableDiffusionPipeline,
        DiffusionPipeline,
        UNet2DConditionModel,
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        EulerDiscreteScheduler
    )
    from diffusers.utils import logging
    from transformers import CLIPTextModel, CLIPTokenizer
    from diffusers.models.autoencoder_kl import AutoencoderKL
    DIFFUSERS_AVAILABLE = True
except ImportError as e:
    print(f"Please install diffusers: pip install diffusers transformers accelerate")
    DIFFUSERS_AVAILABLE = False
    raise e

from .integration import WatermarkUNetWrapper


class WatermarkStableDiffusionPipeline:
    """
    Enhanced Stable Diffusion Pipeline with integrated watermark functionality
    """
    
    def __init__(
        self, 
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        watermark_strength: float = 0.1,
        safety_checker: bool = False
    ):
        self.device = device
        self.torch_dtype = torch_dtype
        self.watermark_strength = watermark_strength
        self.model_path = model_path
        
        print(f"Loading watermarked Stable Diffusion model from {model_path}")
        
        # Load the base pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            safety_checker=None if not safety_checker else "default",
            requires_safety_checker=safety_checker,
            use_safetensors=True
        )
        
        self.pipe = self.pipe.to(device)
        
        # Setup watermark components
        self._setup_watermark_components()
        
        # Setup random number generator for reproducible watermarks
        self.rng = np.random.RandomState(42)
        
        print(f"Pipeline initialized with watermark support: {self.has_watermark}")
        
    def _setup_watermark_components(self):
        """Extract and setup watermark components from the loaded model"""
        state_dict = self.pipe.unet.state_dict()
        
        # Check for watermark components
        watermark_decoder_keys = [k for k in state_dict.keys() if k.startswith('watermark_decoder.')]
        watermark_embed_keys = [k for k in state_dict.keys() if k.startswith('watermark_embed.')]
        
        if not watermark_decoder_keys or not watermark_embed_keys:
            print("Warning: No watermark components found in the model")
            self.has_watermark = False
            return
        
        print(f"Found watermark components:")
        print(f"  Decoder parameters: {len(watermark_decoder_keys)}")
        print(f"  Embedding parameters: {len(watermark_embed_keys)}")
        
        self.has_watermark = True
        
        # Extract decoder parameters
        decoder_state_dict = {}
        for key in watermark_decoder_keys:
            new_key = key.replace('watermark_decoder.', '')
            decoder_state_dict[new_key] = state_dict[key]
        
        # Extract embedding parameters
        embed_state_dict = {}
        for key in watermark_embed_keys:
            new_key = key.replace('watermark_embed.', '')
            embed_state_dict[new_key] = state_dict[key]
        
        # Build decoder from state dict
        self.watermark_decoder = self._build_decoder_from_state_dict(decoder_state_dict)
        self.watermark_decoder = self.watermark_decoder.to(self.device, self.torch_dtype)
        
        # Build embedding from state dict
        self.watermark_embed = self._build_embed_from_state_dict(embed_state_dict)
        self.watermark_embed = self.watermark_embed.to(self.device, self.torch_dtype)
        
        # Determine fingerprint size from decoder
        if 'dense.2.weight' in decoder_state_dict:
            self.fingerprint_size = decoder_state_dict['dense.2.weight'].shape[0]
        else:
            self.fingerprint_size = 48  # Default
        
        print(f"Watermark components loaded successfully")
        print(f"  Fingerprint size: {self.fingerprint_size}")
    
    def _build_decoder_from_state_dict(self, decoder_state_dict: Dict[str, torch.Tensor]):
        """Reconstruct watermark decoder from state dict"""
        try:
            from stegmodel.model import StegDecoder
        except ImportError:
            raise ImportError("Please ensure the stegmodel package is available")
        
        # Infer parameters from state dict
        # Find the final linear layer to determine fingerprint size
        final_weight_key = None
        for key in decoder_state_dict.keys():
            if 'dense' in key and 'weight' in key:
                layer_num = int(key.split('.')[1])
                if final_weight_key is None or layer_num > int(final_weight_key.split('.')[1]):
                    final_weight_key = key
        
        if final_weight_key:
            fingerprint_size = decoder_state_dict[final_weight_key].shape[0]
        else:
            fingerprint_size = 48  # Default
        
        # Infer resolution from conv layers
        first_conv_weight = None
        for key in decoder_state_dict.keys():
            if 'decoder.0.weight' in key:
                first_conv_weight = decoder_state_dict[key]
                break
        
        if first_conv_weight is not None:
            image_channels = first_conv_weight.shape[1]
        else:
            image_channels = 3  # Default
        
        # Create decoder with inferred parameters
        decoder = StegDecoder(
            resolution=512,  # Will be adjusted based on actual usage
            image_channels=image_channels,
            fingerprint_size=fingerprint_size
        )
        
        decoder.load_state_dict(decoder_state_dict)
        return decoder
    
    def _build_embed_from_state_dict(self, embed_state_dict: Dict[str, torch.Tensor]):
        """Reconstruct watermark embedding from state dict"""
        # Determine layer sizes from state dict
        layers = []
        layer_indices = set()
        
        for key in embed_state_dict.keys():
            if 'weight' in key:
                layer_idx = int(key.split('.')[0])
                layer_indices.add(layer_idx)
        
        layer_indices = sorted(layer_indices)
        
        for i, layer_idx in enumerate(layer_indices):
            weight_key = f"{layer_idx}.weight"
            if weight_key in embed_state_dict:
                weight = embed_state_dict[weight_key]
                out_features, in_features = weight.shape
                
                layers.append(nn.Linear(in_features, out_features))
                
                # Add SiLU activation between layers (except last)
                if i < len(layer_indices) - 1:
                    layers.append(nn.SiLU())
        
        embed = nn.Sequential(*layers)
        embed.load_state_dict(embed_state_dict)
        return embed
    
    def generate_watermark(self, batch_size: int = 1, seed: Optional[int] = None) -> torch.Tensor:
        """Generate a random watermark"""
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            return torch.randn(
                batch_size, self.fingerprint_size, 
                generator=generator, device=self.device, dtype=self.torch_dtype
            )
        else:
            return torch.randn(
                batch_size, self.fingerprint_size, 
                device=self.device, dtype=self.torch_dtype
            )
    
    def generate_deterministic_watermark(
        self, 
        seed: int, 
        batch_size: int = 1
    ) -> torch.Tensor:
        """Generate a deterministic watermark from seed"""
        rng = np.random.RandomState(seed)
        watermark = rng.randn(batch_size, self.fingerprint_size).astype(np.float32)
        return torch.from_numpy(watermark).to(self.device, self.torch_dtype)
    
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        watermark: Optional[torch.Tensor] = None,
        watermark_seed: Optional[int] = None,
        return_dict: bool = True,
        callback = None,
        callback_steps: int = 1,
        **kwargs
    ):
        """Generate images with optional watermark embedding"""
        
        # Determine batch size
        if isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        else:
            batch_size = len(prompt)
        
        batch_size *= num_images_per_prompt
        
        # Generate or use provided watermark
        if watermark is not None:
            if watermark.shape[0] != batch_size:
                watermark = watermark.repeat(batch_size, 1)
        elif watermark_seed is not None:
            watermark = self.generate_deterministic_watermark(watermark_seed, batch_size)
        elif self.has_watermark:
            watermark = self.generate_watermark(batch_size)
        
        # Store original UNet forward method
        original_unet_forward = self.pipe.unet.forward
        
        # Apply watermark integration if available
        if watermark is not None and self.has_watermark:
            def watermarked_forward(sample, timestep, encoder_hidden_states, **forward_kwargs):
                # Embed watermark
                wm_emb = self.watermark_embed(watermark)
                
                # Get time embeddings
                if sample.dtype != timestep.dtype:
                    timestep = timestep.to(sample.dtype)
                
                # Get the UNet's time projection
                t_emb = self.pipe.unet.time_proj(timestep)
                t_emb = t_emb.to(dtype=sample.dtype)
                t_emb = self.pipe.unet.time_embedding(t_emb)
                
                # Add watermark embedding
                t_emb = t_emb + self.watermark_strength * wm_emb
                
                # Continue with normal UNet forward pass but with modified time embedding
                return self._forward_unet_with_modified_time_emb(
                    sample, timestep, encoder_hidden_states, t_emb, **forward_kwargs
                )
            
            # Replace UNet forward method
            self.pipe.unet.forward = watermarked_forward
        
        try:
            # Generate images
            result = self.pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                eta=eta,
                generator=generator,
                return_dict=return_dict,
                callback=callback,
                callback_steps=callback_steps,
                **kwargs
            )
            
            # Store watermark info for later verification
            if hasattr(result, 'images') and watermark is not None:
                if return_dict:
                    result.watermark = watermark.cpu()
                else:
                    # If return_dict is False, result is just the images list
                    setattr(result, 'watermark', watermark.cpu())
            
            return result
            
        finally:
            # Restore original UNet forward method
            self.pipe.unet.forward = original_unet_forward
    
    def _forward_unet_with_modified_time_emb(self, sample, timestep, encoder_hidden_states, modified_time_emb, **kwargs):
        """Forward pass through UNet with modified time embedding"""
        # This method manually implements the UNet forward pass with our modified time embedding
        # Note: This is a simplified version and might need adjustment for specific UNet architectures
        
        # Get the UNet model
        unet = self.pipe.unet
        
        # Sample preprocessing
        sample = unet.conv_in(sample)
        
        # Time embedding (already computed and modified)
        time_emb = modified_time_emb
        
        # Text embedding
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states
        
        # Down sampling
        down_block_res_samples = (sample,)
        for downsample_block in unet.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=time_emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=time_emb)
            
            down_block_res_samples += res_samples
        
        # Mid block
        if unet.mid_block is not None:
            sample = unet.mid_block(
                sample,
                time_emb,
                encoder_hidden_states=encoder_hidden_states,
            )
        
        # Up sampling
        for i, upsample_block in enumerate(unet.up_blocks):
            is_final_block = i == len(unet.up_blocks) - 1
            
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            
            if not is_final_block and hasattr(unet, "upsample_size"):
                upsample_size = unet.upsample_size
            else:
                upsample_size = None
            
            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=time_emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=time_emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )
        
        # Post-process
        if unet.conv_norm_out:
            sample = unet.conv_norm_out(sample)
            sample = unet.conv_act(sample)
        sample = unet.conv_out(sample)
        
        return sample
    
    def extract_watermark(self, images: Union[Image.Image, List[Image.Image], torch.Tensor]) -> torch.Tensor:
        """Extract watermark from generated images"""
        if not self.has_watermark:
            raise ValueError("Watermark decoder not available")
        
        # Convert images to tensor if needed
        if isinstance(images, Image.Image):
            images = [images]
        
        if isinstance(images, list):
            image_tensors = []
            for img in images:
                if isinstance(img, Image.Image):
                    # Convert PIL to tensor
                    img_array = np.array(img).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                    # Normalize to [-1, 1] range like training
                    img_tensor = img_tensor * 2.0 - 1.0
                    image_tensors.append(img_tensor)
                else:
                    image_tensors.append(img)
            images = torch.stack(image_tensors)
        
        images = images.to(self.device, self.torch_dtype)
        
        # Extract watermarks
        with torch.no_grad():
            extracted_watermarks = self.watermark_decoder(images)
        
        return extracted_watermarks
    
    def verify_watermark(
        self, 
        images: Union[Image.Image, List[Image.Image], torch.Tensor],
        original_watermark: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Verify if images contain the expected watermark"""
        extracted_watermarks = self.extract_watermark(images)
        
        # Ensure original_watermark is on the same device
        original_watermark = original_watermark.to(extracted_watermarks.device)
        
        # Compute similarity
        similarities = []
        for i in range(extracted_watermarks.shape[0]):
            extracted_wm = extracted_watermarks[i:i+1]
            if original_watermark.shape[0] > i:
                orig_wm = original_watermark[i:i+1]
            else:
                orig_wm = original_watermark[0:1]
            
            sim = F.cosine_similarity(extracted_wm, orig_wm, dim=1).item()
            similarities.append(sim)
        
        # Check if verification passes
        verification_results = [sim > threshold for sim in similarities]
        
        return {
            'similarities': similarities,
            'verification_passed': verification_results,
            'average_similarity': float(np.mean(similarities)),
            'threshold': threshold,
            'pass_rate': float(np.mean(verification_results))
        }
    
    def save_config(self, config_path: str):
        """Save pipeline configuration"""
        config = {
            'model_path': str(self.model_path),
            'watermark_strength': self.watermark_strength,
            'device': self.device,
            'torch_dtype': str(self.torch_dtype),
            'has_watermark': self.has_watermark,
            'fingerprint_size': getattr(self, 'fingerprint_size', 48)
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Pipeline configuration saved to {config_path}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        **kwargs
    ):
        """Load a watermarked pipeline from pretrained model"""
        return cls(
            model_path=model_path,
            device=device,
            torch_dtype=torch_dtype,
            **kwargs
        )
    
    def enable_memory_efficient_attention(self):
        """Enable memory efficient attention if available"""
        if hasattr(self.pipe, 'enable_memory_efficient_attention'):
            self.pipe.enable_memory_efficient_attention()
        return self
    
    def enable_xformers_memory_efficient_attention(self):
        """Enable xformers memory efficient attention if available"""
        if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(f"Failed to enable xformers: {e}")
        return self
    
    def enable_model_cpu_offload(self):
        """Enable model CPU offloading to save GPU memory"""
        if hasattr(self.pipe, 'enable_model_cpu_offload'):
            self.pipe.enable_model_cpu_offload()
        return self
