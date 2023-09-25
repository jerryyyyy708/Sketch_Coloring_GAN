from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import os
import matplotlib.pyplot as plt


class Sketch_Data(Dataset):
    def __init__(self, root, shrink = True):
        self.root = root
        self.imgs = os.listdir(self.root)
        self.shrink = shrink
        self.transform_colored = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform_gray = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.5,), (0.5,))
        ])
    
    def __getitem__(self, index):
        # Open the image and convert to RGB
        img_file = Image.open(os.path.join(self.root, self.imgs[index])).convert("RGB")
        
        # Resize the image to half its original dimensions
        if self.shrink:
            width, height = img_file.size
            img_file = img_file.resize((width // 2, height // 2))

        new_width, new_height = img_file.size
        colored = img_file.crop((0, 0, new_width // 2, new_height))
        sketch = img_file.crop((new_width // 2, 0, new_width, new_height))
        
        sketch = sketch.convert("L")
        
        return self.transform_gray(sketch), self.transform_colored(colored)
    
    def __len__(self):
        return len(self.imgs)

    def plot_sample(self, index):
        left, right_gray = self.__getitem__(index)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
        ax1.imshow(left)
        ax1.set_title('Sketch')
        ax2.imshow(right_gray, cmap='gray')
        ax2.set_title('Grayscale Colored Image')
        plt.show()