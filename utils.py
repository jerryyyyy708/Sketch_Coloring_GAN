from PIL import Image
from torchvision.transforms import ToPILImage
import torch
import torch.nn as nn
import os

def save_example(epoch, output, x, y, idx = 0, output_dir = 'output_images'):
    # Saving model's output
    transform_to_image = ToPILImage()
    sample_output = output[idx].cpu()
    img_output = transform_to_image(sample_output)
    img_output.save(os.path.join(output_dir, f'epoch_{epoch+1}_output.png'))
    
    # Saving original sketch
    sample_sketch = x[idx].cpu()
    img_sketch = transform_to_image(sample_sketch)
    img_sketch.save(os.path.join(output_dir, f'epoch_{epoch+1}_sketch.png'))
    
    # Saving original colored image
    sample_image = y[idx].cpu()
    img_image = transform_to_image(sample_image)
    img_image.save(os.path.join(output_dir, f'epoch_{epoch+1}_image.png'))