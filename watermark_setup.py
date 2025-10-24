import onnxruntime as ort
import os 
import sys

sys.path.append("./Infinity")
sys.path.append("./lm-watermarking")

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
import pandas as pd
from sklearn import metrics
import re
from trustmark import TrustMark

from helper import set_seeds
import watermark_attacks
import argparse

random.seed(42)  # For reproducibility

MODEL_PATH = "/path/watermark_comparison/models"


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, num_samples=-1, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.image_names = []
        
        # Collect all image paths from subdirectories
            
        for file in os.listdir(root_dir):
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                self.image_paths.append(os.path.join(root_dir, file))
                self.image_names.append(file)
                
        if num_samples >0:
            if num_samples < len(self.image_paths):
                self.image_paths = self.image_paths[:num_samples]
                self.image_names = self.image_names[:num_samples]
                
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, self.image_names[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, self.image_names[idx]

class Rivagan:
    def __init__(self):
        providers = [
    ('CUDAExecutionProvider', {'device_id': 0}),  # Specify the GPU device ID
    'CPUExecutionProvider']
        self.encoder =  ort.InferenceSession(f'{MODEL_PATH}/rivagan_encoder.onnx', providers=providers)
        self.decoder =  ort.InferenceSession(f'{MODEL_PATH}/rivagan_decoder.onnx', providers=providers)

    def encode(self, images, messages):
        images = torch.stack([transforms.ToTensor()(img) for img in images])
        images = (images - 0.5) * 2
        images = torch.clamp(images,-1.0, 1.0)
        images = images.unsqueeze(2)
        inputs = {
            'frame': images.detach().cpu().numpy(),
            'data': messages.detach().cpu().numpy(),
        }
        wm_images = np.stack(self.encoder.run(None, inputs))
        wm_images = torch.clamp(torch.from_numpy(wm_images), min=-1.0, max=1.0)
        wm_images = (wm_images / 2) + 0.5
        wm_images = wm_images.squeeze()

        to_pil = transforms.ToPILImage(mode='RGB')
        return [to_pil(wm_images)]

    def decode(self, images):
        images = torch.stack([transforms.ToTensor()(img) for img in images])
        images = (images - 0.5) * 2
        images = torch.clamp(images,-1.0, 1.0)
        images = images.unsqueeze(2)
        inputs = {
            'frame': images.detach().cpu().numpy(),
        }
        outputs = self.decoder.run(None, inputs)
        messages = outputs[0]
        return messages


class StegaStamp:
    def __init__(self):
        providers = [
    ('CUDAExecutionProvider', {'device_id': 0}),  # Specify the GPU device ID
    'CPUExecutionProvider']
        self.model =  ort.InferenceSession(f'{MODEL_PATH}/stega_stamp.onnx', providers=providers)
        self.resize_down = transforms.Resize((400, 400))
        self.resize_up = transforms.Resize((512, 512))
        
    def encode(self, images, messages):
        images = torch.stack([transforms.ToTensor()(img) for img in images])
        inputs = {
            'image': self.resize_down(images).permute(0, 2, 3, 1).detach().cpu().float().numpy(),
            'secret': messages.detach().cpu().numpy(),
        }
        wm_images = np.stack(self.model.run(None, inputs)[0])
        wm_images = torch.from_numpy(wm_images)
        wm_images = wm_images.permute(0, 3, 1, 2)
        wm_images = self.resize_up(wm_images)
        to_pil = transforms.ToPILImage()
        return [to_pil(wm_images[i])  for i in range(wm_images.size(0))]

    def decode(self, images):
        images = torch.stack([transforms.ToTensor()(img) for img in images])
        inputs = {
            'image': self.resize_down(images).permute(0, 2, 3, 1).detach().cpu().float().numpy(),
            "secret": np.zeros((len(images), 100), dtype=np.float32)
        }
        outputs = self.model.run(None, inputs)
        messages = outputs[2]
        return messages

def inspect_onnx_operations(model_path: str):
    """
    Simple function to inspect ONNX model operations
    
    Args:
        model_path: Path to ONNX model file
    """
    session = ort.InferenceSession(model_path)
    
    print("\nInputs:")
    for input in session.get_inputs():
        print(f"- Name: {input.name}, Shape: {input.shape}, Type: {input.type}")
    
    print("\nOutputs:")
    for output in session.get_outputs():
        print(f"- Name: {output.name}, Shape: {output.shape}, Type: {output.type}")
    
    print("\nProviders:", session.get_providers())


def watermark_images(model_name, image_path, message, num_images, args):
    match model_name:
        case "rivagan":
            model = Rivagan()
            out_path = os.path.join(args.output_path, "rivagan")
            os.makedirs(out_path, exist_ok=True)
        case "stega_stamp":
            model = StegaStamp()
            out_path = os.path.join(args.output_path, "stega_stamp")
            os.makedirs(out_path, exist_ok=True)

        case "trustmark":
            model = TrustMark(verbose=True, model_type='Q')
            out_path = os.path.join(args.output_path, "trustmark")
            os.makedirs(out_path, exist_ok=True)

        case _:
            raise ValueError(f"Unknown model name: {model_name}")
    
    #load images from image_path
    image_data = ImageFolderDataset(image_path, num_samples=num_images)
    
    
    for image, img_name in tqdm(image_data):

        if image: 
            if not os.path.exists(os.path.join(out_path, f"{img_name}")):

                if model_name == "trustmark":
                    wmarked_image = model.encode(image, message)
                else:
                    wmarked_image = model.encode([image], message)[0]
                
                wmarked_image.save(os.path.join(out_path, f"{img_name}"))
                
            else: 
                print(f"Image {img_name} already exists in {out_path}, skipping saving.", flush=True)
        else:
            print(f"Image {img_name} could not be loaded, skipping.", flush=True)
        
    with open(os.path.join(out_path, "message.txt"), "w") as f:
        f.write(f"{message}\n")
    
    print(f"Message has been saved to {out_path}/message.txt")
    

def detect_watermark(model_name, args):
    match model_name:
        case "rivagan":
            model = Rivagan()
            with open(os.path.join(args.output_path, "rivagan", "message.txt"), "r") as f:
                message = f.read().strip()
            
            bits = [int(x) for x in re.findall(r'[01]', message)]
            print(bits)
            
            folder_path = os.path.join(args.watermarked_folder_path, "rivagan")
            
        case "stega_stamp":
            model = StegaStamp()
            with open(os.path.join(args.output_path, "stega_stamp", "message.txt"), "r") as f:
                message = f.read().strip()
            bits = [int(x) for x in re.findall(r'[01]', message)]
            print(bits)
            
            folder_path = os.path.join(args.watermarked_folder_path, "stega_stamp")
        case "trustmark":
            model = TrustMark(verbose=True, model_type='Q')
            secret_str = 'mysecret'
            bits = model.ecc.encode_text([secret_str])[0]
            print(bits)
            folder_path = os.path.join(args.watermarked_folder_path, "trustmark")
            
        
        case _:
            raise ValueError(f"Unknown model name: {model_name}")
    
    clean_dataset = watermark_attacks.apply_attack(args.clean_folder_path, "none", args)
    
    
    all_attacks = ["vertical_flip", "horizontal_flip"] #"none", "noise" , "gauss", "color", "rotate", "crop", "jpeg", "conventional_all", "VAE", "CtrlRegen""vertical_flip", "horizontal_flip"
    
    
    clean_zscores = []
    clean_labels = []
    
    for i, image in enumerate(tqdm(clean_dataset, miniters=50)):

        if i == 0: 
            os.makedirs(f"{folder_path}/attacks", exist_ok=True)
            image.save(f"{folder_path}/attacks/clean.png")
        
        if model_name == "trustmark":
            wm_secret, wm_present, wm_schema = model.decode(image)
            reconstructed_message = model.ecc.encode_text([wm_secret])[0]
        else:
            reconstructed_message = [int(i) for i in model.decode([image])[0]>0.5]
        
        
        assert len(reconstructed_message) == len(bits), f"Decoded message length {len(reconstructed_message)} does not match original message length {len(bits)}"
        
        bit_overlap = sum([1 for i in range(len(reconstructed_message)) if reconstructed_message[i] == int(bits[i])])/len(reconstructed_message)

        if bit_overlap < 0.5: 
            bit_overlap = 1 - bit_overlap  # Account for bit flips
        
        clean_zscores.append(bit_overlap)
        clean_labels.append(0)        
        
    
    for attack in tqdm(all_attacks):

        wmarked_z_scores = []
        wmarked_labels = []
    
        attacked_dataset = watermark_attacks.apply_attack(folder_path, attack, args)
        
        for i, image in enumerate(tqdm(attacked_dataset, miniters=50)):
            
            

            if i == 0: 
                image.save(f"{folder_path}/attacks/{attack}.png")
            
            
            if model_name == "trustmark":
                wm_secret, wm_present, wm_schema = model.decode(image)
                reconstructed_message = model.ecc.encode_text([wm_secret])[0]
            else:
                reconstructed_message = [int(i) for i in model.decode([image])[0]>0.5]
            
  
            bit_overlap = sum([1 for i in range(len(reconstructed_message)) if reconstructed_message[i] == int(bits[i])])/len(reconstructed_message)
            
            if bit_overlap < 0.5: 
                bit_overlap = 1 - bit_overlap  # Account for bit flips
        
        
            wmarked_z_scores.append(bit_overlap)
            wmarked_labels.append(1)
            
            
        all_labels = clean_labels + wmarked_labels
        all_scores = clean_zscores + wmarked_z_scores
    
        fpr, tpr, threshold = metrics.roc_curve(all_labels, all_scores)
        auc = metrics.auc(fpr, tpr)
        acc = np.max(1 - (fpr + (1 - tpr))/2)
        print(f"Threshold: {threshold[np.where(fpr<.01)[0][-1]]}")
        low = tpr[np.where(fpr<.01)[0][-1]]
        
        print(f'For attack {attack}: auc: {auc}, acc: {acc}, TPR@1%FPR: {low}')
        
        df = pd.DataFrame([{
            "Attack": attack,
            "AUC": auc,
            "Accuracy": acc,
            "TPR@1%FPR": low,
            "bit_overlaps": round(np.mean(wmarked_z_scores).item(),4),
            "bit_overlaps_std": round(np.std(wmarked_z_scores).item(),4),
            "clean_bit_overlaps": round(np.mean(clean_zscores).item(),4),
            "clean_bit_overlaps_std": round(np.std(clean_zscores).item(),4)
        }])

        print(df)
        file_path = f"{folder_path}/robustness_results.csv"
        write_header = not os.path.exists(file_path)

        df.to_csv(file_path, mode='a', header=write_header, index=False)

    



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # add_common_arguments(parser)
    parser.add_argument("--model_path", type=str, default=f"{MODEL_PATH}")
    parser.add_argument("--image_path", type=str, default="/path/scales_0/delta_0")
    parser.add_argument("--model_name", type=str, default="rivagan", choices=["rivagan", "stega_stamp", "trustmark"])
    parser.add_argument("--output_path", type=str, default="/path/watermark_comparison")
    parser.add_argument("--message_length", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--clean_folder_path", type=str, default="/path/scales_0/delta_0")
    parser.add_argument("--watermarked_folder_path", type=str, default="/path/watermark_comparison")
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--variance", type=float, default=0.1)
    parser.add_argument("--crop_ratio",  type=float, default=0.7)
    parser.add_argument("--final_quality", type=int, default=75)
    parser.add_argument("--ctrl_regen_steps", type=float, default=0.5)
    parser.add_argument("--stable_diff_vae", type=str, default='')
    parser.add_argument("--setting", type=str, default="robustness", choices=["watermarking", "robustness"])
    parser.add_argument("--mini", default=-30)
    parser.add_argument("--maxi", default=30)
    parser.add_argument("--model_folder_path", type=str, default="/path/models")
    parser.add_argument("--color_jitter_strength", type=float, default=1.0)
    parser.add_argument("--load_message", action='store_true', help="Load message from file instead of generating a new one")
    parser.add_argument("--encoder_robustness", type=int, default=0)

    
    args = parser.parse_args()
    
    set_seeds(42)  # For reproducibility
    
    match args.setting:
        case "watermarking":
            # Generate a random binary message
            if not args.load_message:
                message = torch.randint(0, 2, (1, args.message_length)).float()  
                
                print(f"Generated message: {message}")
            else: 
                with open(os.path.join(args.output_path, args.model_name, "message.txt"), "r") as f:
                    message = f.read().strip()
                    message = torch.tensor([int(i) for i in re.findall(r'[01]', message)]).float().unsqueeze(0)
                    print(f"Loaded message: {message}")

            if args.model_name == "trustmark":
                message = "mysecret"

            with open(os.path.join(args.output_path, "message.txt"), "w") as f:
                f.write(f"{message}\n")

            watermark_images(args.model_name, args.image_path, message, args.num_samples, args)
        
        case "robustness":
            print(f"Detecting watermark for model: {args.model_name}")
            
            detect_watermark(args.model_name, args)
    