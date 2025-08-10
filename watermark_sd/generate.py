#!/usr/bin/env python3
"""
Image generation script using watermarked Stable Diffusion models
"""

import torch
import argparse
import json
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Optional, Union
import time

from .pipeline import WatermarkStableDiffusionPipeline


def parse_prompts(prompt_input: str) -> List[str]:
    """Parse prompt input - can be a single prompt or path to file with prompts"""
    prompt_path = Path(prompt_input)
    
    if prompt_path.exists():
        # Read prompts from file
        with open(prompt_path, 'r', encoding='utf-8') as f:
            if prompt_path.suffix == '.json':
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'prompts' in data:
                    return data['prompts']
                else:
                    return [str(data)]
            else:
                # Text file, one prompt per line
                prompts = [line.strip() for line in f if line.strip()]
                return prompts
    else:
        # Single prompt string
        return [prompt_input]


def generate_images(
    model_path: str,
    prompts: List[str],
    output_dir: str,
    watermark: Optional[torch.Tensor] = None,
    watermark_seed: Optional[int] = None,
    width: int = 512,
    height: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[str] = None,
    num_images_per_prompt: int = 1,
    batch_size: int = 1,
    device: str = "auto",
    dtype: str = "float16",
    seed: Optional[int] = None,
    scheduler: str = "ddim",
    enable_memory_efficient: bool = True
):
    """
    Generate images using watermarked Stable Diffusion pipeline
    """
    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup dtype
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    
    print(f"Using device: {device}")
    print(f"Using dtype: {torch_dtype}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize pipeline
    print(f"Loading pipeline from {model_path}...")
    pipe = WatermarkStableDiffusionPipeline.from_pretrained(
        model_path,
        device=device,
        torch_dtype=torch_dtype
    )
    
    # Configure scheduler if specified
    if scheduler.lower() != "default":
        from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler
        
        if scheduler.lower() == "ddim":
            pipe.pipe.scheduler = DDIMScheduler.from_config(pipe.pipe.scheduler.config)
        elif scheduler.lower() == "dpm":
            pipe.pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.pipe.scheduler.config)
        elif scheduler.lower() == "euler":
            pipe.pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.pipe.scheduler.config)
        
        print(f"Using scheduler: {scheduler}")
    
    # Enable memory optimizations
    if enable_memory_efficient:
        pipe.enable_memory_efficient_attention()
        pipe.enable_xformers_memory_efficient_attention()
        if device == "cuda":
            pipe.enable_model_cpu_offload()
    
    # Setup generator for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        print(f"Using seed: {seed}")
    
    # Generate watermark if needed
    if watermark is None and watermark_seed is not None:
        print(f"Generating watermark with seed: {watermark_seed}")
    elif watermark is not None:
        print(f"Using provided watermark of shape: {watermark.shape}")
    elif pipe.has_watermark:
        print("Generating random watermark")
    else:
        print("No watermark will be embedded (model doesn't support it)")
    
    # Process prompts in batches
    all_results = []
    total_images = 0
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_idx = i // batch_size
        
        print(f"\nProcessing batch {batch_idx + 1}/{(len(prompts) + batch_size - 1) // batch_size}")
        print(f"Prompts in batch: {len(batch_prompts)}")
        
        start_time = time.time()
        
        # Generate images for this batch
        result = pipe(
            prompt=batch_prompts,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            watermark=watermark,
            watermark_seed=watermark_seed
        )
        
        generation_time = time.time() - start_time
        print(f"Generation time: {generation_time:.2f}s ({generation_time/len(result.images):.2f}s per image)")
        
        # Save images
        for j, image in enumerate(result.images):
            prompt_idx = i + (j // num_images_per_prompt)
            image_idx = j % num_images_per_prompt
            
            # Create safe filename from prompt
            safe_prompt = "".join(c for c in batch_prompts[prompt_idx][:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_prompt = safe_prompt.replace(' ', '_')
            
            filename = f"{total_images:04d}_{safe_prompt}_{image_idx:02d}.png"
            image_path = output_dir / filename
            
            image.save(image_path)
            print(f"Saved: {filename}")
            total_images += 1
        
        # Store results for verification
        batch_result = {
            'batch_idx': batch_idx,
            'prompts': batch_prompts,
            'num_images': len(result.images),
            'generation_time': generation_time,
            'watermark': result.watermark.cpu().numpy().tolist() if hasattr(result, 'watermark') else None
        }
        all_results.append(batch_result)
    
    # Save generation metadata
    metadata = {
        'model_path': model_path,
        'total_images': total_images,
        'parameters': {
            'width': width,
            'height': height,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'negative_prompt': negative_prompt,
            'num_images_per_prompt': num_images_per_prompt,
            'seed': seed,
            'watermark_seed': watermark_seed,
            'scheduler': scheduler
        },
        'device': device,
        'dtype': str(torch_dtype),
        'batch_results': all_results
    }
    
    metadata_path = output_dir / "generation_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nGeneration completed!")
    print(f"Total images generated: {total_images}")
    print(f"Images saved to: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")
    
    return all_results, metadata


def verify_watermarks(
    model_path: str,
    images_dir: str,
    metadata_path: str,
    threshold: float = 0.5,
    device: str = "auto"
):
    """
    Verify watermarks in generated images
    """
    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load pipeline
    pipe = WatermarkStableDiffusionPipeline.from_pretrained(model_path, device=device)
    
    if not pipe.has_watermark:
        print("Model does not support watermark extraction")
        return
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    images_dir = Path(images_dir)
    verification_results = []
    
    for batch_result in metadata['batch_results']:
        if batch_result['watermark'] is None:
            continue
        
        # Convert watermark back to tensor
        original_watermark = torch.tensor(batch_result['watermark'], device=device)
        
        batch_idx = batch_result['batch_idx']
        num_images = batch_result['num_images']
        
        # Find images for this batch
        batch_images = []
        for i in range(num_images):
            # This is a simplified way to find images - you might need to adjust based on your naming scheme
            image_files = list(images_dir.glob(f"*batch_{batch_idx:04d}*.png"))
            if i < len(image_files):
                batch_images.append(Image.open(image_files[i]))
        
        if not batch_images:
            continue
        
        # Verify watermarks
        verification = pipe.verify_watermark(batch_images, original_watermark, threshold)
        
        verification_results.append({
            'batch_idx': batch_idx,
            'num_images': len(batch_images),
            'verification': verification
        })
        
        print(f"Batch {batch_idx}: {verification['pass_rate']:.2%} pass rate "
              f"(avg similarity: {verification['average_similarity']:.3f})")
    
    # Overall statistics
    if verification_results:
        all_similarities = []
        all_passed = []
        
        for result in verification_results:
            all_similarities.extend(result['verification']['similarities'])
            all_passed.extend(result['verification']['verification_passed'])
        
        overall_pass_rate = np.mean(all_passed)
        overall_similarity = np.mean(all_similarities)
        
        print(f"\nOverall verification results:")
        print(f"  Pass rate: {overall_pass_rate:.2%}")
        print(f"  Average similarity: {overall_similarity:.3f}")
        print(f"  Threshold: {threshold}")
    
    return verification_results


def main():
    parser = argparse.ArgumentParser(description="Generate images with watermarked Stable Diffusion")
    
    # Required arguments
    parser.add_argument("--model_path", required=True, help="Path to watermarked Stable Diffusion model")
    parser.add_argument("--prompt", required=True, help="Text prompt or path to prompts file")
    parser.add_argument("--output_dir", default="./generated_images", help="Output directory")
    
    # Generation parameters
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images per prompt")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--negative_prompt", help="Negative prompt")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing prompts")
    
    # Watermark parameters
    parser.add_argument("--watermark_seed", type=int, help="Seed for deterministic watermark")
    parser.add_argument("--no_watermark", action='store_true', help="Disable watermark embedding")
    
    # Technical parameters
    parser.add_argument("--device", default="auto", help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16", help="Model dtype")
    parser.add_argument("--seed", type=int, help="Random seed for generation")
    parser.add_argument("--scheduler", choices=["default", "ddim", "dpm", "euler"], default="ddim", help="Scheduler type")
    parser.add_argument("--disable_memory_efficient", action='store_true', help="Disable memory optimizations")
    
    # Verification
    parser.add_argument("--verify", action='store_true', help="Verify watermarks after generation")
    parser.add_argument("--verify_threshold", type=float, default=0.5, help="Verification threshold")
    
    args = parser.parse_args()
    
    # Parse prompts
    prompts = parse_prompts(args.prompt)
    print(f"Found {len(prompts)} prompts")
    
    # Generate images
    results, metadata = generate_images(
        model_path=args.model_path,
        prompts=prompts,
        output_dir=args.output_dir,
        watermark_seed=None if args.no_watermark else args.watermark_seed,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        num_images_per_prompt=args.num_images,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        scheduler=args.scheduler,
        enable_memory_efficient=not args.disable_memory_efficient
    )
    
    # Verify watermarks if requested
    if args.verify and not args.no_watermark:
        print("\nVerifying watermarks...")
        verify_watermarks(
            model_path=args.model_path,
            images_dir=args.output_dir,
            metadata_path=Path(args.output_dir) / "generation_metadata.json",
            threshold=args.verify_threshold,
            device=args.device
        )


if __name__ == "__main__":
    main()
