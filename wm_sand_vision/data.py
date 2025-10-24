import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import json
import numpy as np
import matplotlib.pyplot as plt

class WatermarkedImageDataset(Dataset):
    def __init__(self, image_folder='imgs', json_path='metrics.json', scheme='invisible-watermark', sample_size=None, transform=None):
        """
        Args:
            image_folder (string): folder with all the images.
            json_path (string): path to the metrics.json file containing prompts and image_ids.
            scheme (string): watermarking scheme used.
            sample_size (int, optional): Number of samples to use. If None, use all samples.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_folder = image_folder
        self.transform = transform
        self.scheme = scheme
        
        # Load image files
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
        
        # Load metrics.json
        with open(json_path, 'r') as f:
            # The file might contain multiple JSON objects on separate lines
            json_data = []
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        json_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Handle potential incomplete JSON objects
                        pass
        
        # Create mapping from image_id to prompt
        self.samples = []
        for item in json_data:
            image_id = item.get('image_id')
            prompt = item.get('prompt')
            if image_id and prompt:
                image_filename = f"{image_id}.png"
                # Check if the image file exists
                if image_filename in self.image_files:
                    self.samples.append({
                        'image_id': image_id,
                        'image_path': os.path.join(image_folder, image_filename),
                        'prompt': prompt
                    })
        
        # Limit sample size if specified
        if sample_size is not None and sample_size < len(self.samples):
            self.samples = self.samples[:sample_size]
            
        print(f"Loaded {len(self.samples)} valid image-prompt pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample['image_path']
        prompt = sample['prompt']
        image_id = sample['image_id']
        
        # Load the image if needed (commented out as the original code only returned paths)
        # image = Image.open(image_path).convert('RGB')
        # if self.transform:
        #     image = self.transform(image)
        
        return {'image': image_path, 'prompt': prompt, 'image_id': image_id}


def test_watermarked_image_dataset():
    """Test the WatermarkedImageDataset class with actual data."""
    # Define paths
    base_path = '/path/wm_sand/scales_2/delta_3'
    image_folder = base_path
    json_path = os.path.join(base_path, 'metrics.json')
    
    # Create dataset instance
    dataset = WatermarkedImageDataset(
        image_folder=image_folder,
        json_path=json_path,
        sample_size=None  # Use all samples
    )
    
    # Basic checks
    print(f"Dataset size: {len(dataset)}")
    
    # Check a few samples
    num_samples_to_check = min(5, len(dataset))
    for i in range(num_samples_to_check):
        sample = dataset[i]
        
        # Check image path
        image_path = sample['image']
        assert os.path.exists(image_path), f"Image path does not exist: {image_path}"
        
        # Check prompt
        prompt = sample['prompt']
        assert prompt and isinstance(prompt, str), f"Invalid prompt: {prompt}"
        
        print(f"Sample {i}:")
        print(f"  Image path: {image_path}")
        print(f"  Prompt: {prompt}")
        
        # Try loading the image (optional)
        try:
            img = Image.open(image_path)
            print(f"  Image size: {img.size}")
        except Exception as e:
            print(f"  Error loading image: {e}")
    
    # Test with DataLoader
    print("\nTesting with DataLoader...")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(dataloader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Batch size: {len(batch['image'])}")
    
    # Visualize a few samples (optional)
    visualize_samples(dataset, num_samples=2)
    
    print("\nAll tests passed successfully!")


def visualize_samples(dataset, num_samples=2):
    """Visualize a few samples from the dataset."""
    plt.figure(figsize=(12, 6 * num_samples))
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        img_path = sample['image']
        prompt = sample['prompt']
        
        # Load and display image
        img = Image.open(img_path)
        plt.subplot(num_samples, 1, i + 1)
        plt.imshow(img)
        plt.title(f"Prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_visualization.png')
    print("Visualization saved to 'sample_visualization.png'")


if __name__ == "__main__":
    test_watermarked_image_dataset()