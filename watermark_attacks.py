from huggingface_hub import __getattr__
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from diffusers import AutoencoderKL, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import CannyDetector 
from transformers import AutoModel, AutoImageProcessor
import numpy as np
from random import random, shuffle

import sys

sys.path.append("./Infinity")
sys.path.append("./lm-watermarking")

from PIL import Image
from Infinity.CtrlRegen.custom_i2i_pipeline import CustomStableDiffusionControlNetImg2ImgPipeline
from color_matcher import ColorMatcher
from color_matcher.normalizer import Normalizer
from tqdm import tqdm
from torchvision import transforms

def color_match(ref_img, src_img):
    cm = ColorMatcher() 
    img_ref_np = Normalizer(np.asarray(ref_img)).type_norm()
    img_src_np = Normalizer(np.asarray(src_img)).type_norm()

    img_res = cm.transfer(src=img_src_np, ref=img_ref_np, method='hm-mkl-hm')   # hm-mvgd-hm / hm-mkl-hm
    img_res = Normalizer(img_res).uint8_norm()
    img_res = Image.fromarray(img_res)
    return img_res



import os

import torchvision.transforms.v2


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, num_samples=-1, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        # Collect all image paths from subdirectories
        
        # if os.path.isdir(root_dir):
        #     for subdir in os.listdir(root_dir):
        #         subdir_path = os.path.join(root_dir, subdir)
        #         if os.path.isdir(subdir_path):
        #             for file in os.listdir(subdir_path):
        #                 if file.lower().endswith(('png', 'jpg', 'jpeg')):
        #                     self.image_paths.append(os.path.join(subdir_path, file))
            
        for file in os.listdir(root_dir):
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                self.image_paths.append(os.path.join(root_dir, file))
        shuffle(self.image_paths)
        if num_samples >0:
            if num_samples < len(self.image_paths):
                self.image_paths = self.image_paths[:num_samples]
        

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__(idx-1)
        if self.transform:
            image = self.transform(image)
        
        return image

class EncoderRobustnessDataset(ImageFolderDataset):
    def __init__(self, root_dir, num_samples=-1, transform=None):
        super().__init__(root_dir, num_samples, transform)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            gen_bits = torch.load(img_path.replace('.png', '_gen_bits.pt'))
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__(idx-1)
        
        if self.transform:
            image = self.transform(image)
        return image, gen_bits
    
    
def add_gaussian_noise(image, mean=0.0, std=10.0):
    """
    Add Gaussian noise to a PIL image.

    Args:
        image (PIL.Image): Input image.
        mean (float): Mean of Gaussian noise.
        std (float): Standard deviation of Gaussian noise.

    Returns:
        PIL.Image: Image with added Gaussian noise.
    """
    # Convert PIL image to NumPy array
    img_array = np.array(image).astype(np.float32)
    
    # Add Gaussian noise
    noise = np.random.normal(mean, std, img_array.shape)
    noisy_array = img_array + noise

    # Clip values to valid range and convert back to uint8
    noisy_array = np.clip(noisy_array, 0, 255).astype(np.uint8)

    # Convert back to PIL image
    return Image.fromarray(noisy_array)

class TensorImageDataset(Dataset):
    def __init__(self, tensor_array, transform=None):
        """
        tensor_array: a list or tensor of shape (N, C, H, W)
        transform: optional transform to be applied on a sample
        """
        self.tensor_array = tensor_array
        self.transform = transforms.ToPILImage()

    def __len__(self):
        return len(self.tensor_array)

    def __getitem__(self, idx):
        # image, gen_bits = self.tensor_array[idx]
        image = self.tensor_array[idx]
        if self.transform:
            image = self.transform(image)
        return image
        # return image, gen_bits


#Define Attacks
#Conventional
def noise(dataset, args):

    variance = args.variance 
    assert variance >= 0 and variance <= 1, "Variance must be between 0 and 1"

    noise_added = 255  * variance


    transform = transforms.Compose([
        #transforms.ToTensor(),
        transforms.Lambda(lambda x: add_gaussian_noise(x, mean=0.0, std=noise_added)),
        #transforms.ToTensor(),
    ])
    
    #apply gaussian noise with variance 
    
    #apply gaussian blur with 8x8 filter
    
    dataset.transform = transform
    
    return dataset

def gauss(dataset, args):

    kernel_size = args.kernel_size
    variance = args.variance
    assert variance >= 0 and variance <= 1, "Variance must be between 0 and 1"

    noise_added = 255  * variance

    transform = transforms.Compose([
        #transforms.Lambda(lambda x: add_gaussian_noise(x, mean=0.0, std=noise_added)),
        transforms.GaussianBlur(kernel_size),
        #transforms.ToTensor(),
    ])
    
    #apply gaussian noise with variance 
    
    #apply gaussian blur with 8x8 filter
    
    dataset.transform = transform
    
    return dataset

def color(dataset, args, hue=0.5, saturation=5.0, contrast=5.0, brightness=5.0):
    
    #apply color jitter with random hue (0.3)
    #saturation scaling (3.0)
    #contrast scaling (3.0)
    color_jitter_strength = args.color_jitter_strength
    hue = hue * color_jitter_strength
    saturation = saturation * color_jitter_strength
    contrast = contrast * color_jitter_strength
    brightness = brightness * color_jitter_strength


    transform = transforms.Compose([
        transforms.ColorJitter(hue=(hue,hue), saturation=(saturation,saturation), contrast=(contrast,contrast), brightness=(brightness,brightness)),
        #transforms.ToTensor(),
    ])
    
    dataset.transform = transform
    
    return dataset

def crop(dataset, args):
    
    #crop and resize: 0.7, random rotation 0-180degrees
    crop_ratio = args.crop_ratio

    img = dataset[0]
    original_size = img.size[0]
    
    transform = transforms.Compose([
        transforms.RandomCrop(size=crop_ratio*original_size),
        # transforms.RandomRotation((0, 180)),
        transforms.Resize((original_size, original_size)),
        #transforms.ToTensor(),
    ])
    
    dataset.transform = transform
    
    return dataset

def rotate(dataset, args):
    
    transform = transforms.Compose([
        transforms.RandomRotation(args.rotate_degrees),
    ])
    
    dataset.transform = transform
    
    return dataset

def horizontal_flip(dataset, args):
    
    transform = transforms.RandomHorizontalFlip(p=1)
    
    dataset.transform= transform
    
    return dataset

def vertical_flip(dataset, args):
    
    transform = transforms.RandomVerticalFlip(p=1)
    
    dataset.transform= transform
    
    return dataset

def jpeg(dataset, args, compression=0.25):

    final_quality = args.final_quality
    assert 1 <= final_quality <= 100, "Quality must be between 1 and 100"
    
    transform = transforms.Compose([
        torchvision.transforms.v2.JPEG(final_quality), #25% compression
        #transforms.ToTensor()
    ])
    
    dataset.transform = transform
    
    return dataset

def conventional_all(dataset, args):
    img = dataset[0]
    original_size = img.size[0]
    
    transform = transforms.Compose([
        transforms.Lambda(lambda x: add_gaussian_noise(x, mean=0.0, std=25)),
        transforms.GaussianBlur(7),
        transforms.ColorJitter(hue=0.3, saturation=3.0, contrast=3.0),
        transforms.RandomCrop(size=0.7*original_size),
        transforms.RandomRotation((0, 180)),
        transforms.Resize((original_size, original_size)),
        torchvision.transforms.v2.JPEG(75),
        #transforms.ToTensor()    
    ])
    
    dataset.transform = transform
    
    return dataset


#Regeneration
def VAE(dataset, args): #vae_path as kwarg
    
    #encoding and deconding with vae of StableDiffusion 1.5 or 2.1
    
    vae_path = args.stable_diff_vae
    
    vae = AutoencoderKL.from_pretrained(vae_path).to("cuda")

    t = transforms.ToTensor()
    
    dataset.transform = t

    def encode_img(input_img):
        # Single image -> single latent in a batch (so size 1, 4, 64, 64)
        if len(input_img.shape)<4:
            input_img = input_img.unsqueeze(0)
        with torch.no_grad():
            latent = vae.encode(input_img*2 - 1) # Note scaling
        return 0.18215 * latent.latent_dist.sample()


    def decode_img(latents):
        # bath of latents -> list of images
        latents = (1 / 0.18215) * latents
        with torch.no_grad():
            image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach()
        return image
    
    processed = [(decode_img(encode_img(image.to("cuda"))).squeeze(0).to("cpu")) for image in dataset]
    #processed = [(decode_img(encode_img(image.to("cuda"))).squeeze(0).to("cpu"), gen_bits) for image, gen_bits in dataset]
    
    dataset = TensorImageDataset(processed)
    
    return dataset

def DiffPure(dataset, args, t=0.15):
    
    #https://arxiv.org/abs/2408.11039
    
    raise NotImplementedError("This is not yet implemented")
    
    return dataset

def CtrlRegen(dataset, args): 


    #https://github.com/yepengliu/ctrlregen
    
    device =  'cuda'

    transform_size_to_512 = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            ])
    BASE_PATH = args.model_folder_path
    DIFFUSION_MODEL = 'SG161222/Realistic_Vision_V4.0_noVAE'
    SPATIAL_CONTROL_PATH = BASE_PATH + '/ctrlregen/spatialnet_ckp/spatial_control_ckp_14000'
    SEMANTIC_CONTROL_PATH = BASE_PATH + '/ctrlregen/semanticnet_ckp'
    SEMANTIC_CONTROL_NAME = BASE_PATH + '/ctrlregen/semanticnet_ckp/models/semantic_control_ckp_435000.bin'
    IMAGE_ENCODER = 'facebook/dinov2-giant'
    VAE = 'stabilityai/sd-vae-ft-mse'

    spatialnet = [ControlNetModel.from_pretrained(SPATIAL_CONTROL_PATH, torch_dtype=torch.float16)]
    pipe = CustomStableDiffusionControlNetImg2ImgPipeline.from_pretrained(DIFFUSION_MODEL, \
                                                            controlnet=spatialnet, \
                                                            torch_dtype=torch.float16,
                                                            safety_checker = None,
                                                            requires_safety_checker = False
                                                            )
    pipe.costum_load_ip_adapter(SEMANTIC_CONTROL_PATH, subfolder='models', weight_name=SEMANTIC_CONTROL_NAME)
    pipe.image_encoder = AutoModel.from_pretrained(IMAGE_ENCODER, cache_dir=BASE_PATH).to(device, dtype=torch.float16)
    pipe.feature_extractor = AutoImageProcessor.from_pretrained(IMAGE_ENCODER, cache_dir=BASE_PATH)
    pipe.vae = AutoencoderKL.from_pretrained(VAE, cache_dir=BASE_PATH).to(dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_ip_adapter_scale(1.0)
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)

    processor = CannyDetector()

    def ctrl_regen_plus(input_img, step, seed=0):
        generator = torch.manual_seed(seed)
        input_img = transform_size_to_512(input_img)
        processed_img = processor(input_img, low_threshold=100, high_threshold=150)
        prompt = 'best quality, high quality'
        negative_prompt = 'monochrome, lowres, bad anatomy, worst quality, low quality'
        output_img = pipe(prompt,
                        negative_prompt=negative_prompt,
                        image = [input_img],
                        control_image = [processed_img], # spatial condition
                        ip_adapter_image = [input_img],   # semantic condition
                        strength = step,
                        generator = generator,
                        num_inference_steps=50,
                        controlnet_conditioning_scale = 1.0,
                        guidance_scale = 2.0,
                        control_guidance_start = 0,
                        control_guidance_end = 1,
                        ).images[0]
        output_img = color_match(input_img, output_img)
        return output_img
    img = dataset[0]
    org_size = img.size[0]
    t = transforms.Compose([
        transforms.Resize(org_size),
        transforms.ToTensor(),
        
    ])

    num_steps = args.ctrl_regen_steps

    #processed = [(t(ctrl_regen_plus(image, step=num_steps)), gen_bits) for i, (image, gen_bits) in enumerate(tqdm(dataset)) if i <= args.num_samples]
    processed = [(t(ctrl_regen_plus(image, step=num_steps))) for i, (image) in enumerate(tqdm(dataset)) if i <= args.num_samples]

    dataset = TensorImageDataset(processed)
    
    return dataset

def none(dataset, args):
    return dataset


def apply_attack(img_path, attack, args):

    if args.encoder_robustness == 1:
        dataset = EncoderRobustnessDataset(img_path, num_samples=args.num_samples)    
    else:

        dataset = ImageFolderDataset(img_path, num_samples=args.num_samples)

    
    attack_map = {
        'noise' : noise,
        'gauss': gauss,
        'color': color,
        'crop': crop,
        'rotate' : rotate,
        'jpeg' : jpeg,
        'VAE' : VAE, #vae_path as kwarg
        'DiffPure': DiffPure,
        'CtrlRegen' : CtrlRegen,
        'horizontal_flip' : horizontal_flip,
        'vertical_flip' : vertical_flip,
        'none' : none,
    }


    if attack not in attack_map:
        raise ValueError(f"Unsupported attack: {attack}")
    
    
    return attack_map[attack](dataset, args)

