import argparse
import os
import json
import os
import torch
import torch_fidelity
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import logging
import pandas as pd
from helper import set_seeds
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import numpy as np
from datasets import GeneratedImageDataset, get_matched_nat_dataset, ImageOnlyDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(encoding="utf-8", level=logging.WARNING)
logger = logging.getLogger(__name__)
# Compute FID, KID, and ISC using torch-fidelity
def calculate_metrics(generated_dataset, matched_nat_dataset):
    print(f"Start FID calculation")

    kid_subset_size = min(1000,min(len(matched_nat_dataset), len(generated_dataset)))
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=generated_dataset,
        input2=matched_nat_dataset,
        cuda=torch.cuda.is_available(),  # Automatically use GPU if available
        isc=True,     # Inception Score (ISC)
        fid=True,     # Fréchet Inception Distance (FID)
        kid=True,     # Kernel Inception Distance (KID)
        prc=True,    # Precision/Recall 
        verbose=False, # Enable verbose mode
        kid_subset_size=kid_subset_size,  # Subset size for KID calculation
        samples_find_deep=True,
        cache=False,
        no_class=True
    )
    return metrics_dict

    
def infer_fid(nat_data_path, json_gen_path):

    print(f"Currently investigating {'/'.join(json_gen_path.split('/')[:-1])}")

    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).clamp(0, 255).to(torch.uint8))
    ]) # Transform according to https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/feature_extractor_inceptionv3.py


    matched_nat_dataset = ImageOnlyDataset(get_matched_nat_dataset(nat_data_path, json_gen_path, transform))
    
    generated_dataset = ImageOnlyDataset(GeneratedImageDataset(json_gen_path, transform))
    return calculate_metrics(generated_dataset, matched_nat_dataset)

def infer_clip(json_gen_path, clip_model):
    clip_score_fn = partial(clip_score, model_name_or_path=clip_model)
    batch_size = 256
    clip_scores = []
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),    
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # CLIP normalization
    ])
    
    dataset = GeneratedImageDataset(json_gen_path, transform=transform)
    dl = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True, persistent_workers=True)
    
    clip_score_summed = 0
    num_samples = 0
    
    print(f"Processing {len(dataset)} samples with batch size {batch_size}")
    
    for i, (imgs, prompts) in enumerate(tqdm(dl)):
        try:
            imgs = imgs.to(device, non_blocking=True)
            
            with torch.no_grad():  
                batch_score = calculate_clip_score(clip_score_fn, imgs, list(prompts))
            
            clip_scores.append(batch_score)
            clip_score_summed += batch_score * len(prompts)
            num_samples += len(prompts)
        except Exception as e:
            print(f"Error processing batch {i} with error {e}")
            
    print(f"CLIP score: {clip_scores}")

    if num_samples:
        clip_score_mean = clip_score_summed / num_samples
        std = np.std(clip_scores)
        std = round(std.item() if hasattr(std, 'item') else std, 3)
    else:
        clip_score_mean = clip_score_summed
        std = -1
    return clip_score_mean, std  


def calculate_clip_score(clip_score_fn, images, prompts):
    if images.device != device:
        images = images.to(device)
    
    clip_score_result = clip_score_fn(images, prompts)
    if hasattr(clip_score_result, 'detach'):
        clip_score_result = clip_score_result.detach()
    
    return round(float(clip_score_result), 4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nat_data_path", type=str)
    parser.add_argument("--gen_data_path", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=None, help="Override default batch size")
    
    args = parser.parse_args()
    set_seeds(args.seed)
    
    nat_data_path = args.nat_data_path
    gen_data_path = args.gen_data_path
    json_gen_path = gen_data_path if gen_data_path.endswith('.json') else f"{gen_data_path}/metrics.json"
    long_clip = "zer0int/LongCLIP-L-Diffusers" 
    clip_score_mean, clip_score_std = infer_clip(json_gen_path, long_clip)
    metrics_dict = infer_fid(nat_data_path, json_gen_path)
    print("Computing Long CLIP scores...")
    metrics_dict['long_clip_score_std'] = clip_score_std
    metrics_dict['long_clip_score_mean'] = clip_score_mean
    
    print("Computing standard CLIP scores...")
    short_clip = "openai/clip-vit-large-patch14"
    clip_score_mean, clip_score_std = infer_clip(json_gen_path, short_clip)

    metrics_dict['clip_score_std'] = clip_score_std
    metrics_dict['clip_score_mean'] = clip_score_mean
    df = pd.DataFrame(metrics_dict, index=[0])
    df.to_csv(f"{args.out_dir}/img_quality.csv", mode='w', index=False)
    
    print(f"Results saved to {args.out_dir}/img_quality.csv")
    print("Metrics computed:")
    for key, value in metrics_dict.items():
        print(f"  {key}: {value}")
    
    


