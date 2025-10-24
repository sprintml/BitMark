import watermark_attacks

import json
import logging
logging.basicConfig(encoding="utf-8", level=logging.WARNING)
logger = logging.getLogger(__name__)
import torch
from tqdm import tqdm

import detect_watermark
from detect_watermark import get_detector
from helper import set_seeds
torch._dynamo.config.cache_size_limit = 64

import pandas as pd
import os
import torch
from sklearn import metrics
import numpy as np
from architecture_wrapper import VAEWrapper, get_architecture_arguments, get_vae



def analyze_rotation(watermarked_folder_path, clean_folder_path, vae_wrapper, watermark_detector, watermark_scales, args):
    clean_dataset = watermark_attacks.apply_attack(clean_folder_path, "none", args)    
    
    clean_zscores = []
    clean_labels = []
    
    for i, image in enumerate(tqdm(clean_dataset, miniters=50)):

        detect_results = detect_watermark.detect(args, image, watermark_detector=watermark_detector, vae_wrapper=vae_wrapper, watermark_scales=watermark_scales, detect_on_each_scale=True)
        
        clean_zscores.append(detect_results["z_score"])
        clean_labels.append(0)

    for degree in tqdm(range(0, 360, 10), miniters=50):
    
        args.mini = degree
        args.maxi = degree
    
        wmarked_z_scores = []
        wmarked_labels = []
    
        attacked_dataset = watermark_attacks.apply_attack(watermarked_folder_path, "rotate", args)
        
        for i, image in enumerate(tqdm(attacked_dataset, miniters=50)):

            detect_results = detect_watermark.detect(args, image, watermark_detector=watermark_detector, vae_wrapper=vae_wrapper, watermark_scales=watermark_scales, detect_on_each_scale=True)
            
            wmarked_z_scores.append(detect_results["z_score"])
            wmarked_labels.append(1)
                
        all_labels = clean_labels + wmarked_labels
        all_scores = clean_zscores + wmarked_z_scores
    
        fpr, tpr, threshold = metrics.roc_curve(all_labels, all_scores)
        auc = metrics.auc(fpr, tpr)
        acc = np.max(1 - (fpr + (1 - tpr))/2)
        print(f"Threshold: {threshold[np.where(fpr<.01)[0][-1]]}")
        low = tpr[np.where(fpr<.01)[0][-1]]
        
        print(f'For rotation {degree}: auc: {auc}, acc: {acc}, TPR@1%FPR: {low}')
        
        df = pd.DataFrame([{
            "Scale" : watermark_scales,
            "Degree": degree,
            "AUC": auc,
            "Accuracy": acc,
            "TPR@1%FPR": low,
            "z_score": round(np.mean(wmarked_z_scores).item(),4),
            "z_score_std": round(np.std(wmarked_z_scores).item(),4),
            "clean_z_score": round(np.mean(clean_zscores).item(),4),
            "clean_z_score_std": round(np.std(clean_zscores).item(),4)
        }])

        file_path = f"{watermarked_folder_path}/rotation_results.csv"
        write_header = not os.path.exists(file_path)

        df.to_csv(file_path, mode='a', header=write_header, index=False)


def extended_analysis(watermarked_folder_path, clean_folder_path, vae_wrapper, watermark_detector, watermark_scales, args):
    
    clean_dataset = watermark_attacks.apply_attack(clean_folder_path, "none", args)    
    
    clean_zscores = []
    clean_labels = []
    
    for i, image in enumerate(tqdm(clean_dataset, miniters=50)):
        
        detect_results = detect_watermark.detect(args, image, watermark_detector=watermark_detector, vae_wrapper=vae_wrapper, watermark_scales=watermark_scales, detect_on_each_scale=True)

        #print(f"Green fraction for clean: {watermark_metrics["green_fraction"]}", flush=True)
        clean_zscores.append(detect_results["z_score"])
        clean_labels.append(0)

    all_attacks = [ "gauss", "noise" , "color", "rotate", "crop", "jpeg", "VAE", "CtrlRegen"]  #, CtrlRegen, DiffPure "noise", "gauss", "crop", "jpeg", 
    range_map = {
        "gauss": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
        "noise": [0.0, 0.05, 0.1, 0.15, 0.2],
        "color": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "rotate": [0, 10, 20, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 310, 320, 330, 340, 350],
        "crop": [1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5],
        "jpeg": [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
        "CtrlRegen": [0.1, 0.2, 0.3, 0.4, 0.5],
        "VAE":  [0.0]
    }


    args_map = {
        "gauss" : "kernel_size",
        "noise" : "variance",
        "color" : "color_jitter_strength",
        "rotate" : "rotate_degrees",
        "crop" : "crop_ratio",
        "jpeg" : "final_quality",
        "CtrlRegen" : "ctrl_regen_steps",
        "VAE" : "kernel_size"
    }

    for attack in tqdm(all_attacks):
    
        for attack_strength in tqdm(range_map[attack]):

            args.__dict__[args_map[attack]] = attack_strength

            wmarked_z_scores = []
            wmarked_labels = []
        
            attacked_dataset = watermark_attacks.apply_attack(watermarked_folder_path, attack, args)
            
            for i, image in enumerate(tqdm(attacked_dataset, miniters=50)):

                if i == 0: image.save(f"{watermarked_folder_path}/attacks/{attack}_{attack_strength}.png")
                
                detect_results = detect_watermark.detect(args, image, watermark_detector=watermark_detector, vae_wrapper=vae_wrapper, watermark_scales=watermark_scales, detect_on_each_scale=True)

                wmarked_z_scores.append(detect_results["z_score"])
                wmarked_labels.append(1)
                    
            all_labels = clean_labels + wmarked_labels
            all_scores = clean_zscores + wmarked_z_scores
        
            fpr, tpr, threshold = metrics.roc_curve(all_labels, all_scores)
            auc = metrics.auc(fpr, tpr)
            acc = np.max(1 - (fpr + (1 - tpr))/2)
            print(f"Threshold: {threshold[np.where(fpr<.01)[0][-1]]}")
            low = tpr[np.where(fpr<.01)[0][-1]]
            
            print(f'For attack {attack} with strength {attack_strength}: auc: {auc}, acc: {acc}, TPR@1%FPR: {low}')
            
            df = pd.DataFrame([{
                "Scale" : watermark_scales,
                "Attack": attack,
                "AUC": auc,
                "Accuracy": acc,
                "TPR@1%FPR": low,
                "Attack Strength": attack_strength,
                "z_score": round(np.mean(wmarked_z_scores).item(),4),
                "z_score_std": round(np.std(wmarked_z_scores).item(),4),
                "clean_z_score": round(np.mean(clean_zscores).item(),4),
                "clean_z_score_std": round(np.std(clean_zscores).item(),4)
            }])

            file_path = f"{args.watermarked_dir}/robustness_results_{attack}.csv"
            write_header = not os.path.exists(file_path)

            df.to_csv(file_path, mode='a', header=write_header, index=False)

def tpr_fpr_robustness(watermarked_folder_path, clean_folder_path, vae_wrapper, watermark_detector,  args, attack=None):
    watermark_scales = args.watermark_scales


    clean_dataset = watermark_attacks.apply_attack(clean_folder_path, "none", args)
    
    if attack: 
        if type(attack) == str: 
            all_attacks = [attack]
        else:
            all_attacks = attack 
    else:
        all_attacks = ["none", "noise" , "gauss", "color", "rotate", "crop", "jpeg", "conventional_all", "VAE", "CtrlRegen"] #, CtrlRegen, DiffPure
    
    
    clean_zscores = []
    clean_labels = []
    
    for i, image in enumerate(tqdm(clean_dataset, miniters=50)):

        if i == 0: image.save(f"{watermarked_folder_path}/attacks/clean.png")
        detect_results = detect_watermark.detect(args, image, watermark_detector=watermark_detector, vae_wrapper=vae_wrapper, watermark_scales=watermark_scales, detect_on_each_scale=True)

        #print(f"Green fraction for clean: {watermark_metrics["green_fraction"]}", flush=True)
        clean_zscores.append(detect_results["z_score"])
        clean_labels.append(0)


    for attack in tqdm(all_attacks):
    
        wmarked_z_scores = []
        wmarked_labels = []
        wmarked_green_fractions = []

        attacked_dataset = watermark_attacks.apply_attack(watermarked_folder_path, attack, args)
        
        for i, image in enumerate(tqdm(attacked_dataset, miniters=50)):

            if i == 0: image.save(f"{watermarked_folder_path}/attacks/{attack}.png")
            
            detect_results = detect_watermark.detect(args, image, watermark_detector=watermark_detector, vae_wrapper=vae_wrapper, watermark_scales=watermark_scales, detect_on_each_scale=False)

            #print(f"Green fraction for clean: {watermark_metrics["green_fraction"]}", flush=True)
            wmarked_z_scores.append(detect_results["z_score"])
            wmarked_labels.append(1)
            wmarked_green_fractions.append(detect_results["green_fraction"])

        all_labels = clean_labels + wmarked_labels
        all_scores = clean_zscores + wmarked_z_scores
    
        fpr, tpr, threshold = metrics.roc_curve(all_labels, all_scores)
        auc = metrics.auc(fpr, tpr)
        acc = np.max(1 - (fpr + (1 - tpr))/2)
        print(f"Threshold: {threshold[np.where(fpr<.01)[0][-1]]}")
        low = tpr[np.where(fpr<.01)[0][-1]]
        
        print(f'For attack {attack}: auc: {auc}, acc: {acc}, TPR@1%FPR: {low}')
        
        df = pd.DataFrame([{
            "Scale" : watermark_scales,
            "Attack": attack,
            "AUC": auc,
            "Accuracy": acc,
            "TPR@1%FPR": low,
            "green_fraction": round(np.mean(wmarked_green_fractions).item(),5),
            "green_fraction_std": round(np.std(wmarked_green_fractions).item(),5),
            "z_score": round(np.mean(wmarked_z_scores).item(),4),
            "z_score_std": round(np.std(wmarked_z_scores).item(),4),
            "clean_z_score": round(np.mean(clean_zscores).item(),4),
            "clean_z_score_std": round(np.std(clean_zscores).item(),4)
        }])
        file_path = f"{args.watermarked_dir}/robustness_results"
        if args.watermark_remove_duplicates == 1:
            file_path += "_removed_duplicates"
        if args.watermark_add_noise == 1:
            file_path += "_add_noise"
        file_path += f".csv"

        write_header = not os.path.exists(file_path)

        df.to_csv(file_path, mode='a', header=write_header, index=False)
        
def encoder_robustness(folder_path, vae_wrapper, args, attacks):
    for attack in attacks:
        overlap = []
        attacked_dataset = watermark_attacks.apply_attack(folder_path, attack, args)

        for i, (image, gen_bits) in enumerate(tqdm(attacked_dataset, miniters=50)):
            if i == 0: image.save(f"{folder_path}/attacks/{attack}.png")
            
            with torch.no_grad():
                metric = vae_wrapper.calc_bit_overlap(gen_bits, image, 0)
                overlap.append(metric['overlap_ratio'])

        df = pd.DataFrame([{
            "Attack": attack,
            "Overlap Ratio": np.mean(overlap).round(4),
            "Overlap Ratio Std": np.std(overlap).round(4)
        }])
        file_path = f"{folder_path}/robustness_encoder.csv"
        write_header = not os.path.exists(file_path)

        df.to_csv(file_path, mode='a', header=write_header, index=False)

if __name__ == "__main__":
    
    parser = get_architecture_arguments()
    parser.add_argument("--watermarked_dir", type=str)
    parser.add_argument("--clean_dir", type=str)
    parser.add_argument("--stable_diff_vae", type=str, default='')
    parser.add_argument("--num_samples", type=int, default=-1) # on how many samples the attacks are supposed to happen
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--variance", type=float, default=0.1)
    parser.add_argument("--crop_ratio",  type=float, default=0.7)
    parser.add_argument("--final_quality", type=int, default=75)
    parser.add_argument("--ctrl_regen_steps", type=float, default=0.5)
    parser.add_argument("--model_folder_path", type=str, default="")
    parser.add_argument("--color_jitter_strength", type=float, default=0.6)
    parser.add_argument("--rotate_degrees", type=float, default=180)
    parser.add_argument("--encoder_robustness", type=int, default=0, choices=[0,1], help="If set to 1, encoder robustness is tested")
    args = parser.parse_args()


    set_seeds(args.seed)
    
    # load text encoder
    # load vae
    vae_wrapper: VAEWrapper = get_vae(args)
    # load infinity
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    jsonl_list = []
    cnt = 0

    watermark_detector = get_detector(args)
    
    
    os.makedirs(f"{args.watermarked_dir}/attacks", exist_ok=True)
    print(f"Test robustness between clean dir:{args.clean_dir}, and watermarked dir:{args.watermarked_dir} for the watermark scales: {args.watermark_scales}")
    
    #assert f"scales_{args.watermark_scales}" in args.watermarked_dir, f"Scales do not match between selected dir and watermark scales {args.watermark_scales} and {args.watermarked_dir}"    
    #analyze_rotation(watermarked_folder_path=args.watermarked_dir, clean_folder_path=args.clean_dir, vae=vae, watermark_detector=watermark_detector, watermark_scales=watermark_scales, args=args)
    args.mini=0
    args.maxi=180
    
    set_seeds(args.seed)
    attacks = ['none','noise', 'gauss', 'color', 'crop', 'rotate', 'jpeg', 'VAE', 'horizontal_flip', 'vertical_flip', 'CtrlRegen']

    #if args.encoder_robustness == 1:
    #    encoder_robustness(folder_path=args.watermarked_dir, vae_wrapper=vae_wrapper, args=args, attacks=attacks)
    
    tpr_fpr_robustness(watermarked_folder_path=args.watermarked_dir, clean_folder_path=args.clean_dir, vae_wrapper=vae_wrapper, watermark_detector=watermark_detector, args=args, attack=attacks) # runs all attacks
    #extended_analysis(watermarked_folder_path=args.watermarked_dir, clean_folder_path=args.clean_dir, vae=vae, watermark_detector=watermark_detector, watermark_scales=watermark_scales, args=args) # runs all attacks