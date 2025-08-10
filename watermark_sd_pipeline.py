#!/usr/bin/env python3
"""
Pipeline for text-to-image generation with watermark using modified Stable Diffusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Dict, Any
import numpy as np
from PIL import Image
import json
from pathlib import Path

try:
    from diffusers import (
        StableDiffusionPipeline, 
        UNet2DConditionModel,
        DDIMScheduler,
        DPMSolverMultistepScheduler
    )
    from diffusers.utils import logging
    from transformers import CLIPTextModel, CLIPTokenizer
    from diffusers.models.autoencoder_kl import AutoencoderKL
except ImportError as e:
    print(f"Please install diffusers: pip install diffusers transformers accelerate")
    raise e


class WatermarkStableDiffusionPipeline:
    """
    Custom Stable Diffusion Pipeline with integrated watermark functionality
    """
    
    def __init__(
        self, 
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        watermark_strength: float = 0.1
    ):
        self.device = device
        self.torch_dtype = torch_dtype
        self.watermark_strength = watermark_strength
        
        print(f"Loading watermarked Stable Diffusion model from {model_path}")
        
        # Verify watermark if requested
    if args.verify_watermark and hasattr(result, 'watermark'):
        print("\nVerifying watermarks...")
        verification_results = pipe.verify_watermark(result.images, result.watermark)
        
        print(f"Average similarity: {verification_results['average_similarity']:.4f}")
        for i, (sim, passed) in enumerate(zip(verification_results['similarities'], verification_results['verification_passed'])):
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"Image {i}: similarity = {sim:.4f} {status}")
        
        # Save verification results
        verification_path = output_dir / "watermark_verification.json"
        with open(verification_path, 'w') as f:
            json.dump(verification_results, f, indent=2)
        print(f"Verification results saved to {verification_path}")
    
    # Save configuration
    pipe.save_config(output_dir / "pipeline_config.json")
    print(f"\nGeneration completed! Images saved to {output_dir}")


if __name__ == "__main__":
    main() Load the base pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        self.pipe = self.pipe.to(device)
        
        # Extract watermark components from the loaded state dict
        self._setup_watermark_components()
        
        # Setup random number generator for reproducible watermarks
        self.rng = np.random.RandomState(42)
        
    def _setup_watermark_components(self):
        """Extract and setup watermark components from the loaded model"""
        # Check if model has watermark components
        state_dict = self.pipe.unet.state_dict()
        
        watermark_keys = [k for k in state_dict.keys() if 'watermark' in k.lower()]
        
        if not watermark_keys:
            print("Warning: No watermark components found in the model")
            self.has_watermark = False
            return
        
        print(f"Found {len(watermark_keys)} watermark parameters")
        self.has_watermark = True
        
        # Extract watermark decoder
        decoder_keys = [k for k in watermark_keys if 'decoder' in k]
        embed_keys = [k for k in watermark_keys if 'embed' in k]
        
        # Setup watermark decoder
        if decoder_keys:
            self.watermark_decoder = self._build_decoder_from_state_dict(
                {k.replace('watermark_decoder.', ''): state_dict[k] for k in decoder_keys}
            )
            self.watermark_decoder = self.watermark_decoder.to(self.device, self.torch_dtype)
            print("Watermark decoder loaded successfully")
        
        # Setup watermark embedding
        if embed_keys:
            self.watermark_embed = self._build_embed_from_state_dict(
                {k.replace('watermark_embed.', ''): state_dict[k] for k in embed_keys}
            )
            self.watermark_embed = self.watermark_embed.to(self.device, self.torch_dtype)
            print("Watermark embedding loaded successfully")
    
    def _build_decoder_from_state_dict(self, decoder_state_dict: Dict[str, torch.Tensor]):
        """Reconstruct watermark decoder from state dict"""
        # Infer decoder architecture from state dict
        conv_layers = []
        dense_layers = []
        
        for key in sorted(decoder_state_dict.keys()):
            if 'decoder.' in key and 'weight' in key:
                layer_idx = int(key.split('.')[1])
                while len(conv_layers) <= layer_idx:
                    conv_layers.append(None)
                
                weight_shape = decoder_state_dict[key].shape
                if len(weight_shape) == 4:  # Conv layer
                    out_ch, in_ch, kh, kw = weight_shape
                    if layer_idx % 2 == 0:  # Conv layers
                        stride = 2 if layer_idx in [0, 4, 8, 12, 16] else 1
                        conv_layers[layer_idx] = nn.Conv2d(in_ch, out_ch, (kh, kw), stride, 1)
                    else:  # ReLU layers
                        conv_layers[layer_idx] = nn.ReLU()
            
            elif 'dense.' in key and 'weight' in key:
                layer_idx = int(key.split('.')[1])
                while len(dense_layers) <= layer_idx:
                    dense_layers.append(None)
                
                weight_shape = decoder_state_dict[key].shape
                out_features, in_features = weight_shape
                if layer_idx % 2 == 0:  # Linear layers
                    dense_layers[layer_idx] = nn.Linear(in_features, out_features)
                else:  # ReLU layers
                    dense_layers[layer_idx] = nn.ReLU()
        
        # Remove None entries
        conv_layers = [layer for layer in conv_layers if layer is not None]
        dense_layers = [layer for layer in dense_layers if layer is not None]
        
        # Create decoder
        class WatermarkDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = nn.Sequential(*conv_layers)
                self.dense = nn.Sequential(*dense_layers)
            
            def forward(self, x):
                x = self.decoder(x)
                x = x.view(x.size(0), -1)
                return self.dense(x)
        
        decoder = WatermarkDecoder()
        decoder.load_state_dict(decoder_state_dict)
        return decoder
    
    def _build_embed_from_state_dict(self, embed_state_dict: Dict[str, torch.Tensor]):
        """Reconstruct watermark embedding from state dict"""
        layers = []
        
        for key in sorted(embed_state_dict.keys()):
            if 'weight' in key:
                layer_idx = int(key.split('.')[0])
                while len(layers) <= layer_idx:
                    layers.append(None)
                
                weight_shape = embed_state_dict[key].shape
                out_features, in_features = weight_shape
                
                if layer_idx % 2 == 0:  # Linear layers
                    layers[layer_idx] = nn.Linear(in_features, out_features)
                else:  # SiLU layers
                    layers[layer_idx] = nn.SiLU()
        
        # Remove None entries and create embedding
        layers = [layer for layer in layers if layer is not None]
        embed = nn.Sequential(*layers)
        embed.load_state_dict(embed_state_dict)
        return embed
    
    def generate_watermark(self, batch_size: int = 1, fingerprint_size: int = 48) -> torch.Tensor:
        """Generate a random watermark"""
        return torch.randn(batch_size, fingerprint_size, device=self.device, dtype=self.torch_dtype)
    
    def generate_deterministic_watermark(
        self, 
        seed: int, 
        batch_size: int = 1, 
        fingerprint_size: int = 48
    ) -> torch.Tensor:
        """Generate a deterministic watermark from seed"""
        rng = np.random.RandomState(seed)
        watermark = rng.randn(batch_size, fingerprint_size).astype(np.float32)
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
        """Generate images with optional watermark"""
        
        # Determine batch size
        if isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        else:
            batch_size = len(prompt)
        
        # Generate or use provided watermark
        if watermark is not None:
            if watermark.shape[0] != batch_size:
                watermark = watermark.repeat(batch_size, 1)
        elif watermark_seed is not None:
            watermark = self.generate_deterministic_watermark(watermark_seed, batch_size)
        elif self.has_watermark:
            watermark = self.generate_watermark(batch_size)
        
        # Modify UNet forward pass to include watermark
        if watermark is not None and self.has_watermark:
            original_forward = self.pipe.unet.forward
            watermark_embed = self.watermark_embed(watermark)
            
            def watermarked_forward(sample, timestep, encoder_hidden_states, **forward_kwargs):
                # Get original time embedding
                t_emb = self.pipe.unet.time_embed(
                    self.pipe.unet.get_time_embed(timestep, sample.device)
                )
                
                # Add watermark embedding
                t_emb = t_emb + self.watermark_strength * watermark_embed
                
                # Override time embedding temporarily
                original_time_embed = self.pipe.unet.time_embed
                self.pipe.unet.time_embed = lambda x: t_emb
                
                try:
                    result = original_forward(sample, timestep, encoder_hidden_states, **forward_kwargs)
                finally:
                    self.pipe.unet.time_embed = original_time_embed
                
                return result
            
            # Replace forward method
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
            
            # Store watermark info for later extraction
            if hasattr(result, 'images') and watermark is not None:
                result.watermark = watermark.cpu()
            
            return result
            
        finally:
            # Restore original forward method
            if watermark is not None and self.has_watermark:
                self.pipe.unet.forward = original_forward
    
    def extract_watermark(self, images: Union[Image.Image, List[Image.Image], torch.Tensor]) -> torch.Tensor:
        """Extract watermark from generated images"""
        if not self.has_watermark or not hasattr(self, 'watermark_decoder'):
            raise ValueError("Watermark decoder not available")
        
        # Convert images to tensor if needed
        if isinstance(images, Image.Image):
            images = [images]
        
        if isinstance(images, list):
            # Convert PIL images to tensor
            image_tensors = []
            for img in images:
                if isinstance(img, Image.Image):
                    img_array = np.array(img).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
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
        
        # Compute similarity
        similarities = []
        for i in range(extracted_watermarks.shape[0]):
            sim = F.cosine_similarity(
                extracted_watermarks[i:i+1], 
                original_watermark[i:i+1] if original_watermark.shape[0] > i else original_watermark[0:1],
                dim=1
            ).item()
            similarities.append(sim)
        
        # Check if verification passes
        verification_results = [sim > threshold for sim in similarities]
        
        return {
            'similarities': similarities,
            'verification_passed': verification_results,
            'average_similarity': np.mean(similarities),
            'threshold': threshold
        }
    
    def save_config(self, config_path: str):
        """Save pipeline configuration"""
        config = {
            'watermark_strength': self.watermark_strength,
            'device': self.device,
            'torch_dtype': str(self.torch_dtype),
            'has_watermark': self.has_watermark
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved to {config_path}")


def main():
    """Example usage of WatermarkStableDiffusionPipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate images with watermark using Stable Diffusion")
    parser.add_argument("--model_path", required=True, help="Path to watermarked Stable Diffusion model")
    parser.add_argument("--prompt", required=True, help="Text prompt for image generation")
    parser.add_argument("--output_dir", default="./outputs", help="Output directory for generated images")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--watermark_seed", type=int, help="Seed for deterministic watermark")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--verify_watermark", action='store_true', help="Verify watermark extraction")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize pipeline
    pipe = WatermarkStableDiffusionPipeline(
        model_path=args.model_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Generate images
    print(f"Generating {args.num_images} image(s) with prompt: '{args.prompt}'")
    
    result = pipe(
        prompt=args.prompt,
        num_images_per_prompt=args.num_images,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        watermark_seed=args.watermark_seed
    )
    
    # Save images
    for i, image in enumerate(result.images):
        image_path = output_dir / f"generated_image_{i:03d}.png"
        image.save(image_path)
        print(f"Saved: {image_path}")
    
    #
