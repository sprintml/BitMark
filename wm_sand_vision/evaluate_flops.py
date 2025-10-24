import os
import torch
import numpy as np
from PIL import Image
import logging
import time
from tqdm import tqdm
import copy
import json

# Import classes from cv_attack.py
from cv_attack import Trainer, CLIPQualityOracle, QualityOracle
from cv_utils import generate_square_mask, interpolate_img
from data import WatermarkedImageDataset
from torch.utils.data import DataLoader

log = logging.getLogger("flops_evaluation")
logging.basicConfig(level=logging.INFO)

def format_flops(flops):
    """Format flops in human-readable format"""
    if flops < 1e3:
        return f"{flops:.2f} FLOPs"
    elif flops < 1e6:
        return f"{flops/1e3:.2f} KFLOPs"
    elif flops < 1e9:
        return f"{flops/1e6:.2f} MFLOPs"
    elif flops < 1e12:
        return f"{flops/1e9:.2f} GFLOPs"
    else:
        return f"{flops/1e12:.2f} TFLOPs"

class FLOPsCounter:
    def __init__(self, trainer):
        self.trainer = trainer
        self.perturbation_oracle = trainer.perturbation_oracle
        self.quality_oracle = trainer.quality_oracle
        self.total_flops = 0
        self.breakdown = {
            "perturbation_oracle": 0,
            "quality_oracle": 0,
            "mask_generation": 0,
            "interpolation": 0,
            "other": 0
        }

    def estimate_perturbation_oracle_flops(self, prompts, images, mask_images, num_inference_steps, guidance_scale, num_images_per_prompt):
        """
        Estimate FLOPs for the StableDiffusionInpaintPipeline
        Based on research papers estimating SD model complexity
        """
        batch_size = len(prompts)
        
        # UNet complexity - the core of diffusion models
        # UNet has roughly 860M parameters and each parameter typically requires ~2 FLOPs per inference
        unet_params = 860e6  # 860 million parameters
        # Each inference step requires forward pass through UNet
        # Each forward pass does ~2 FLOPs per parameter
        # We multiply by 2 for guidance_scale > 1 (classifier-free guidance uses two passes)
        guidance_factor = 2 if guidance_scale > 1 else 1
        unet_flops_per_step = unet_params * 2 * guidance_factor * batch_size
        
        # Text encoder (CLIP) - processes prompts
        # CLIP text encoder has ~123M parameters
        text_encoder_params = 123e6
        text_encoder_flops = text_encoder_params * 2 * batch_size
        
        # VAE encoder and decoder - processes images
        # VAE has ~84M parameters
        vae_params = 84e6
        # VAE encodes once and decodes once
        vae_flops = vae_params * 4 * batch_size
        
        # Total FLOPs: UNet (per step) * steps + Text encoding + VAE encoding/decoding
        sd_pipeline_flops = (unet_flops_per_step * num_inference_steps) + text_encoder_flops + vae_flops
        
        # Typical SD inpainting at 512x512 with 50 steps is ~4.5-5.5 TFLOPs
        # If our estimation is way off, use this as a sanity check
        sanity_check_flops = 5e12 * (num_inference_steps / 50) * (batch_size / 1)
        
        # Use the larger of our estimate and the sanity check (ensures we don't underestimate)
        return max(sd_pipeline_flops, sanity_check_flops)
    
    def estimate_quality_oracle_flops(self, prompts, wtmk_images, candidate_images):
        """
        Estimate FLOPs for the CLIPQualityOracle
        Based on CLIP model complexity (ViT-based vision encoder + text encoder)
        """
        batch_size = len(prompts)
        
        # CLIP vision encoder (ViT-B/32)
        # ViT-B/32 has ~86M parameters
        vision_params = 86e6
        # Process both watermarked and candidate images (2 sets)
        vision_flops = vision_params * 2 * 2 * batch_size
        
        # CLIP text encoder (shared with vision)
        # Already calculated above: ~123M parameters 
        text_params = 123e6
        text_flops = text_params * 2 * batch_size
        
        # Matrix multiplication for computing similarities
        # Embedding dimension is typically 512 for CLIP
        embedding_dim = 512
        similarity_flops = embedding_dim * batch_size * 2
        
        # Total FLOPs
        total_clip_flops = vision_flops + text_flops + similarity_flops
        
        # Typical CLIP forward pass is ~340 GFLOPs for a batch of 8
        # Scale accordingly for our batch size
        sanity_check_flops = 340e9 * (batch_size / 8) * 2  # *2 because we process two sets of images
        
        return max(total_clip_flops, sanity_check_flops)
    
    def estimate_mask_generation_flops(self, images, mask_ratio):
        """Estimate FLOPs for mask generation"""
        batch_size = len(images)
        avg_image_size = sum([img.width * img.height for img in images]) / batch_size
        
        # Operations for mask generation:
        # 1. Random number generation: ~3 FLOPs per pixel
        # 2. Thresholding: ~1 FLOP per pixel
        # 3. Mask creation and resizing: ~10 FLOPs per pixel
        flops_per_pixel = 14
        
        return avg_image_size * flops_per_pixel * batch_size
    
    def estimate_interpolation_flops(self, images, inpainted_images, mask_images):
        """Estimate FLOPs for interpolation between original and inpainted images"""
        batch_size = len(images)
        avg_image_size = sum([img.width * img.height for img in images]) / batch_size
        
        # Operations for interpolation:
        # 1. Resize operations: ~5 FLOPs per pixel
        # 2. Mask application: ~3 FLOPs per pixel (multiplication and addition)
        # 3. Blending: ~5 FLOPs per pixel
        flops_per_pixel = 13
        
        return avg_image_size * flops_per_pixel * batch_size * 3  # RGB channels
    
    def analyze_attack_flops(self, prompts, wtmk_images, img_ids, num_images_per_prompt=1, 
                            guidance_scale=5, mask_ratio=0.02, num_inference_steps=75, 
                            attack_steps=10, save_interval=5):
        """Analyze FLOPs for the attack process"""
        images = copy.deepcopy(wtmk_images)
        batch_size = len(prompts)
        
        total_flops = 0
        flops_breakdown = {
            "perturbation_oracle": 0,
            "quality_oracle": 0,
            "mask_generation": 0,
            "interpolation": 0,
            "other": 0
        }
        
        # Calculate average image dimensions for reporting
        avg_width = sum([img.width for img in images]) / batch_size
        avg_height = sum([img.height for img in images]) / batch_size
        log.info(f"Processing {batch_size} images with average dimensions {avg_width:.0f}x{avg_height:.0f}")
        
        for attack_step in tqdm(range(attack_steps)):
            step_start_time = time.time()
            
            # Measure mask generation FLOPs
            mask_start_time = time.time()
            mask_images = [generate_square_mask(image, mask_ratio) for image in images]
            flops_mask = self.estimate_mask_generation_flops(images, mask_ratio)
            flops_breakdown["mask_generation"] += flops_mask
            
            # Measure perturbation oracle FLOPs
            perturbation_start_time = time.time()
            inpainted_images = self.perturbation_oracle(
                prompt=prompts, image=images, mask_image=mask_images, 
                num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt
            ).images
            flops_perturbation = self.estimate_perturbation_oracle_flops(
                prompts, images, mask_images, num_inference_steps, guidance_scale, num_images_per_prompt
            )
            flops_breakdown["perturbation_oracle"] += flops_perturbation
            
            # Measure interpolation FLOPs
            interpolation_start_time = time.time()
            candidate_images = [
                interpolate_img(image, inpainted_image.resize((image.width, image.height)), mask_image=mask_image)
                for image, inpainted_image, mask_image in zip(images, inpainted_images, mask_images)
            ]
            flops_interpolation = self.estimate_interpolation_flops(images, inpainted_images, mask_images)
            flops_breakdown["interpolation"] += flops_interpolation
            
            # Measure quality oracle FLOPs
            quality_start_time = time.time()
            wtmk_scores_per_image, candidate_scores_per_image = self.quality_oracle.judge(prompts, wtmk_images, candidate_images)
            flops_quality = self.estimate_quality_oracle_flops(prompts, wtmk_images, candidate_images)
            flops_breakdown["quality_oracle"] += flops_quality
            
            # Other operations - comparison, mask application, etc.
            # These are negligible compared to the neural network operations
            other_flops = batch_size * 100  # Approximation for simple operations
            flops_breakdown["other"] += other_flops
            
            # Calculate better images
            better_image_mask = (wtmk_scores_per_image - candidate_scores_per_image) <= self.quality_oracle.tie_threshold 
            images = [candidate_images[i] if better_image_mask[i] else images[i] for i in range(batch_size)]
            
            step_flops = flops_mask + flops_perturbation + flops_interpolation + flops_quality + other_flops
            total_flops += step_flops
            
            log.info(f"Attack step {attack_step+1}/{attack_steps}, FLOPs: {format_flops(step_flops)}")
            
        self.total_flops = total_flops
        self.breakdown = flops_breakdown
        
        return total_flops, flops_breakdown
    
    def report(self):
        """Generate a report of the FLOPs analysis"""
        total_flops_formatted = format_flops(self.total_flops)
        
        breakdown_formatted = {}
        for key, value in self.breakdown.items():
            breakdown_formatted[key] = {
                "value": format_flops(value),
                "percentage": f"{(value / self.total_flops) * 100:.2f}%"
            }
        
        log.info("=" * 50)
        log.info(f"Total FLOPs: {total_flops_formatted}")
        log.info("Breakdown:")
        for key, value in breakdown_formatted.items():
            log.info(f"  {key}: {value['value']} ({value['percentage']})")
        log.info("=" * 50)
        
        return {
            "total_flops": self.total_flops,
            "total_flops_formatted": total_flops_formatted,
            "breakdown": breakdown_formatted
        }

def evaluate_flops(
    scheme='wm_sand', 
    sample_size=10,  # Reduced for evaluation
    batch_size=2,    # Reduced for evaluation
    attack_steps=10, # Reduced for evaluation
    mask_ratio=0.02,
    model_name_or_path='stabilityai/stable-diffusion-xl-base-1.0',
    base_path='/path/wm_sand/scales_2/delta_3',
    num_inference_steps=50  # Reduced for evaluation
):
    """Evaluate FLOPs for the attack process"""
    log.info("Initializing trainer and FLOPs counter...")
    trainer = Trainer(
        attack_steps=attack_steps, 
        perturbation_model_name_or_path="stabilityai/stable-diffusion-2-inpainting"
    )
    flops_counter = FLOPsCounter(trainer)
    
    log.info("Loading dataset...")
    json_path = os.path.join(base_path, 'metrics.json')
    try:
        dataset = WatermarkedImageDataset(
            image_folder=base_path,
            json_path=json_path,
            sample_size=sample_size  # Use limited samples for evaluation
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    except Exception as e:
        log.warning(f"Error loading dataset: {e}")
        log.info("Using synthetic data for evaluation...")
        # Create synthetic data for testing if dataset loading fails
        class SyntheticDataset:
            def __init__(self, size=10):
                self.size = size
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                # Create synthetic 512x512 image
                img = Image.new('RGB', (512, 512), color=(73, 109, 137))
                return {
                    'image': img,
                    'prompt': f"A synthetic image {idx}",
                    'image_id': f"synthetic_{idx}"
                }
        
        dataset = SyntheticDataset(sample_size)
        # Create a simple batch manually
        batch = {
            'image': [Image.new('RGB', (512, 512), color=(73, 109, 137)) for _ in range(batch_size)],
            'prompt': [f"A synthetic image {i}" for i in range(batch_size)],
            'image_id': [f"synthetic_{i}" for i in range(batch_size)]
        }
        
        log.info(f"Analyzing synthetic batch...")
        total_flops, flops_breakdown = flops_counter.analyze_attack_flops(
            prompts=batch['prompt'],
            wtmk_images=batch['image'],
            img_ids=batch['image_id'],
            num_images_per_prompt=1,
            guidance_scale=5,
            mask_ratio=mask_ratio,
            num_inference_steps=num_inference_steps,
            attack_steps=attack_steps
        )
        
        # Generate report for this batch
        batch_report = flops_counter.report()
        
        # Final analysis
        log.info("FLOPs Evaluation Summary")
        log.info("-" * 30)
        log.info(f"Model: stabilityai/stable-diffusion-2-inpainting")
        log.info(f"Batch size: {batch_size}")
        log.info(f"Attack steps: {attack_steps}")
        log.info(f"Inference steps per attack: {num_inference_steps}")
        log.info(f"Total FLOPs: {batch_report['total_flops_formatted']}")
        
        # Calculate and report extrapolation to full parameters
        full_attack_steps = 200  # Original attack steps
        full_batch_size = 50     # Original batch size
        full_samples = 200       # Original sample size
        full_inference_steps = 100  # Original inference steps
        
        scaling_factor = (full_attack_steps / attack_steps) * (full_inference_steps / num_inference_steps)
        extrapolated_flops = total_flops * scaling_factor
        extrapolated_flops_formatted = format_flops(extrapolated_flops)
        
        log.info(f"Extrapolated FLOPs for full parameters: {extrapolated_flops_formatted}")
        log.info(f"Extrapolated for {full_attack_steps} attack steps and {full_inference_steps} inference steps")
        log.info("-" * 30)
        
        # Save results to JSON file
        results = {
            "batch_report": {
                "total_flops": float(total_flops),  # Convert to float for JSON serialization
                "total_flops_formatted": batch_report['total_flops_formatted'],
                "breakdown": batch_report['breakdown']
            },
            "extrapolated_flops": float(extrapolated_flops),
            "extrapolated_flops_formatted": extrapolated_flops_formatted,
            "parameters": {
                "batch_size": batch_size,
                "attack_steps": attack_steps,
                "inference_steps": num_inference_steps,
                "full_attack_steps": full_attack_steps,
                "full_inference_steps": full_inference_steps
            }
        }
        
        with open("flops_evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        log.info("Results saved to flops_evaluation_results.json")
        
        return results

if __name__ == '__main__':
    evaluate_flops()