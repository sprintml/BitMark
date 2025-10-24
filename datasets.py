
import torch
from torch.utils.data import Dataset
import os
import json
import random
import logging

from PIL import Image
logging.basicConfig(encoding="utf-8", level=logging.WARNING)
logger = logging.getLogger(__name__)

class GenerateCOCO14Dataset(Dataset):
    """
    Generate a PyTorch Dataset for the COCO 2014 dataset.
    """
    def __init__(self, json_path):
        self.metadata = []
        self.json_path = json_path
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.metadata = data['annotations']
        random.shuffle(self.metadata)
 
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        label = self.metadata[idx]
        metadata = {'caption': label['caption'], "prompt_id": label.get('id'), 'image_id': label.get('image_id')}
        return metadata


class GenerateLaionDataset(Dataset):
    """
    Generate a PyTorch Dataset for the Laion dataset.
    """
    def __init__(self, folder_path):
        self.metadata = []
        caption_type = "caption" # caption or Llama_caption
        for img in os.listdir(folder_path):
            img_id = img.split('.')[0]  # Assuming the image file name is the ID
            if img.endswith((".jpg", ".png")):
                json_path = f"{folder_path}/{img_id}.json"
                try:
                    with open(json_path, 'r') as f:
                        label = json.load(f)
                except FileNotFoundError:
                    logger.warning(f"JSON file {json_path} not found, skipping image {img}.")
                    continue
                meta = {'caption': label.get(caption_type), "prompt_id": caption_type, 'image_id': img_id}
                self.metadata.append(meta)
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        return self.metadata[idx]

class GenerateImageNetDataset(Dataset):
    """
    Generate a PyTorch Dataset for the ImageNet dataset.
    """
    def __init__(self, path, num_samples=10000):
        self.metadata = []
        if num_samples <= 1000:
            metadata = range(num_samples)
        else:
            metadata = range(1000) * num_samples // 1000
        for idx, i in enumerate(metadata):
            self.metadata.append({'caption': i, 'prompt_id': idx, 'image_id': f"{i}_{idx}"})
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        return self.metadata[idx]

# Should be possible to combine for multiple datasets
class GeneratedImageDataset(Dataset):
    def __init__(self, metadata_path, transform=None):
        self.transform = transform
        self.image_paths = []
        self.prompts = []
        self._imagenet_classes = None  # Cache for ImageNet class names
        
        with open(metadata_path, mode="r", encoding="utf-8") as json_file:
            for i, line in enumerate(json_file):
                stripped = line.strip()
                if not stripped:
                    print(f"Skipping line {i} as strip did not work. Line: {line}, strip: {line.strip()}")
                    skipped_lines+=1
                    continue
                try:
                    line2 = json.loads(line.strip())
                    del line2["stat_data"]             
                    self.prompts.append(line2["prompt"])
                    if "image_path" in line2.keys():
                        self.image_paths.append(line2['image_path'])
                    else:
                        self.image_paths.append(line2['path'])                  
                except Exception as e:
                    logger.info(e, line2)
                        
    def __len__(self):
        return len(self.image_paths)
    
    def _load_imagenet_classes(self):
            
        txt_file = "imagenet_classes.txt"
        if os.path.exists(txt_file):
            with open(txt_file, 'r') as f:
                self._imagenet_classes = [line.strip() for line in f.readlines()]
            return self._imagenet_classes
        else:
            logger.warning(f"ImageNet classes file '{txt_file}' not found")
            return None
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        prompt = self.prompts[idx]
        # when prompt can converted to int, assume imagenet
        try:
            # If prompt is numeric, it's likely ImageNet class ID
            class_id = int(prompt)
            if self._imagenet_classes is None:
                self._imagenet_classes = self._load_imagenet_classes()
            if class_id < len(self._imagenet_classes):
                prompt = self._imagenet_classes[class_id]
            else:
                prompt = f"ImageNet class {class_id}"
        except (ValueError, TypeError):
            # Keep original prompt if it's not numeric
            pass
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(e)
            logger.warning(f"Could not open image at {img_path}")
            return self.__getitem__(idx-1)
        
        if self.transform:
            image = self.transform(image)
        
        return image, prompt
class ImageNetNatDataset(GeneratedImageDataset):
    """
    A dataset for ImageNet natural images, assuming the generated images are in the same format.
    """
    def __init__(self, imagenet_data_path, gen_metrics_path, transform=None):
        super().__init__(gen_metrics_path, transform=transform)
        self.imagenet_data_path = imagenet_data_path
        self.image_paths = self.get_image_paths(self.image_paths)

    def get_image_paths(self, img_paths):
        matched_paths = []
        for folder in os.listdir(self.imagenet_data_path):
            class_paths = []
            folder_path = os.path.join(self.imagenet_data_path, folder)
            if not os.path.isdir(folder_path):
                continue
            for img in os.listdir(folder_path):
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    class_paths.append(os.path.join(folder_path, img))
            random.shuffle(class_paths)
            matched_paths.extend(class_paths[:len(img_paths)//1000])
        return matched_paths
    
# Given generated images in metrics_path, this returns the corresponding natural COCO images
class COCO2014NatDataset(GeneratedImageDataset):
    def __init__(self,  coco_data_path, gen_metrics_path, transform=None):
        super().__init__(gen_metrics_path, transform=transform)
        self.coco_data_path = coco_data_path
        self.image_paths = self.get_image_paths(self.image_paths)
        

    def get_image_paths(self, img_paths):
        matched_paths = []
        for img_path in img_paths:
            formatted_id = self.format_coco_image_id(img_path)
            matched_path = os.path.join(self.coco_data_path, formatted_id)
            if os.path.exists(matched_path):
                matched_paths.append(matched_path)
            else:
                logger.warning(f"{matched_path} not found")
        return matched_paths

    
    def format_coco_image_id(self, img_path):
        img_path = img_path.split("/")[-1]  
        image_id = int(img_path.split('.')[0])  # Assuming the image file name is the ID
        return f"COCO_val2014_{image_id:012d}.jpg"
    
class LaionNatDataset(GeneratedImageDataset):
    def __init__(self, laion_data_path,gen_metrics_path,  transform=None):
        super().__init__(gen_metrics_path, transform=transform)
        self.laion_data_path = laion_data_path
        self.image_paths = self.get_image_paths(self.image_paths)

    def get_image_paths(self, img_paths):
        matched_paths = []
        for img_path in img_paths:
            img_name = img_path.split("/")[-1]  # Get the image name from the path
            image_path = os.path.join(self.laion_data_path, img_name)
            if os.path.exists(image_path):
                matched_paths.append(image_path)
            else:
                logging.warning(f"Image {img_name} not found in {self.laion_data_path}")
        return matched_paths

class ImageOnlyDataset(Dataset):
    """
    Wrapper to return only images from a dataset that returns (img, prompt) tuples.
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
    def __len__(self):
        return len(self.base_dataset)
    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]
        return img

def get_matched_nat_dataset(nat_data_path, gen_data_path, transform):
    gen_metrics_path = gen_data_path if gen_data_path.endswith('.json') else f"{gen_data_path}/metrics.json"
    if "coco" in nat_data_path:
        nat_data_path = nat_data_path if nat_data_path.endswith('val2014') else f"{nat_data_path}/val2014"
        dataset = COCO2014NatDataset(nat_data_path, gen_metrics_path, transform)
    elif "laion" in nat_data_path:
        dataset = LaionNatDataset(nat_data_path, gen_metrics_path,  transform)
    elif "imagenet" in nat_data_path:
        dataset = ImageNetNatDataset(nat_data_path, gen_metrics_path, transform)
        # For ImageNet, we assume the generated images are in the same format as the natural images
    else:
        raise ValueError(f"Unknown dataset path: {nat_data_path}")
    return dataset

def get_gen_dataset(dataset_path, num_samples=None):
    if "coco" in dataset_path.lower():
        json_path = dataset_path if dataset_path.endswith('.json') else f"{dataset_path}/annotations/captions_val2014.json"
        metadataset = GenerateCOCO14Dataset(json_path)
        infer_type = "val/coco14"
    elif "laion" in dataset_path.lower():
        metadataset = GenerateLaionDataset(dataset_path)
        infer_type = "pop/laion"
    elif "imagenet" in dataset_path.lower():
        metadataset = GenerateImageNetDataset(dataset_path, num_samples=num_samples)
        infer_type = "val/imagenet"
    else:
        raise ValueError(f"Unknown dataset path: {dataset_path}")
    return metadataset, infer_type 