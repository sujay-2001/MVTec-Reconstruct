import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from utils import DenoisingDataset
from model.sadnet import SADNET
from metrics import calculate_psnr, calculate_ssim
import yaml
import collections
import pandas as pd

def main(config):
    
    transform = transforms.Compose([
    transforms.ToTensor(),
    ])
    
    save_path = config['val']['output_save_path']

    val_dataset = DenoisingDataset(root_dir=config['val']['data_path'], mode='val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=config['val']['batch_size'], shuffle=False, num_workers=2,pin_memory=True, drop_last=False)
    
    denoising_model = SADNET(input_channel=3,output_channel=3)
    denoising_model.load_state_dict(torch.load(config['val']['model_checkpoint']))

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    denoising_model.to(device)        
    
    denoising_model.eval()
    val_psnr = {'Overall': 0, 'Defect': 0}
    val_ssim = {'Overall': 0, 'Defect': 0}
    psnr_item = {'Overall': collections.Counter(), 'Defect': collections.Counter()}
    ssim_item = {'Overall': collections.Counter(), 'Defect': collections.Counter()}
    count_item = collections.Counter()
    prev_item = ''
    idx = 0
    for item, clean_images, degraded_images, masks in tqdm(val_loader):
        if item != prev_item:
            idx = 0
        prev_item = item
        clean_images = clean_images.to(device)
        degraded_images = degraded_images.to(device)
        masks = masks.to(device)
        with torch.no_grad():
            inputs = F.interpolate(degraded_images, size=(config['val']['input_size'], config['val']['input_size']), mode='bilinear', align_corners=False)
            outputs = denoising_model(inputs)
            outputs = torch.clamp(outputs, 0, 1)
            if config['val']['interpolate']:
                original_size = clean_images.shape[2:]  # (height, width)
                outputs = F.interpolate(outputs, size=original_size, mode='bilinear', align_corners=False) 
            else:
                clean_images = F.interpolate(clean_images, size=(config['val']['input_size'], config['val']['input_size']), mode='bilinear', align_corners=False)
                masks = F.interpolate(masks, size=(config['val']['input_size'], config['val']['input_size']), mode='bilinear', align_corners=False)     
                degraded_images = F.interpolate(degraded_images, size=(config['val']['input_size'], config['val']['input_size']), mode='bilinear', align_corners=False)                 
            batch_psnr = calculate_psnr(clean_images, outputs, max_pixel_value=1.0)
            batch_ssim = calculate_ssim(clean_images, outputs)
            batch_defect_psnr = calculate_psnr(clean_images*masks, outputs*masks, max_pixel_value=1.0)
            batch_defect_ssim = calculate_ssim(clean_images*masks, outputs*masks)
            val_psnr['Overall'] += batch_psnr * clean_images.size(0)
            val_ssim['Overall'] += batch_ssim * clean_images.size(0)
            val_psnr['Defect'] += batch_defect_psnr * clean_images.size(0)
            val_ssim['Defect'] += batch_defect_ssim * clean_images.size(0)
                        
            psnr_item['Overall'][item] += batch_psnr * clean_images.size(0)
            ssim_item['Overall'][item] += batch_ssim * clean_images.size(0)
            psnr_item['Defect'][item] += batch_defect_psnr * clean_images.size(0)
            ssim_item['Defect'][item] += batch_defect_ssim * clean_images.size(0)
            count_item[item] += 1
            #Save outputs
            to_pil = transforms.ToPILImage()
            outputs = outputs.detach().cpu()
            degraded_images = degraded_images.detach().cpu()
            masks = masks.detach().cpu()
            clean_images = clean_images.detach().cpu()
            output_image = to_pil(outputs[0])
            degraded_image = to_pil(degraded_images[0])
            clean_image = to_pil(clean_images[0])
            mask_image = to_pil(masks[0])
            if not os.path.exists(os.path.join(save_path,item)):
                os.makedirs(os.path.join(save_path,item,'reconstructed'))
                os.makedirs(os.path.join(save_path,item,'degraded'))
                os.makedirs(os.path.join(save_path,item,'clean'))
                os.makedirs(os.path.join(save_path,item,'masks'))
            output_image.save(os.path.join(save_path,item,'reconstructed',f'{idx}.png'))
            degraded_image.save(os.path.join(save_path,item,'degraded',f'{idx}.png'))
            clean_image.save(os.path.join(save_path,item,'clean',f'{idx}.png'))
            mask_image.save(os.path.join(save_path,item,'masks',f'{idx}_mask.png'))
        idx += 1
    val_psnr['Overall'] /= len(val_loader.dataset)
    val_psnr['Defect'] /= len(val_loader.dataset)
    val_ssim['Overall'] /= len(val_loader.dataset)
    val_ssim['Defect'] /= len(val_loader.dataset)
    for item in count_item:
        psnr_item['Overall'][item] /= count_item[item]
        psnr_item['Defect'][item] /= count_item[item]
        ssim_item['Overall'][item] /= count_item[item]
        ssim_item['Defect'][item] /= count_item[item]
        
    print("PSNR:")
    for category in psnr_item:
        print(f"\nCategory: {category}")
        for item, value in psnr_item[category].items():
            print(f"  {item}: {value} dB")

    print("\nSSIM:")
    for category in ssim_item:
        print(f"\nCategory: {category}")
        for item, value in ssim_item[category].items():
            print(f"  {item}: {value}")
    for category in ssim_item:
        print(f"\nCategory: {category}")
        print("Overall PSNR:", val_psnr)
        print("Overall SSIM:", val_ssim)
    
    data = []
    for category in psnr_item:
        for item in psnr_item[category]:
            data.append({
                'Category': category,
                'Item': item,
                'PSNR': psnr_item[category][item],
                'SSIM': ssim_item[category][item]
            })

    # Add overall PSNR and SSIM for each category
    data.append({'Category': 'Overall', 'Item': 'All', 'PSNR': val_psnr['Overall'], 'SSIM': val_ssim['Overall']})
    data.append({'Category': 'Defect', 'Item': 'All', 'PSNR': val_psnr['Defect'], 'SSIM': val_ssim['Defect']})

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Export to Excel
    metrics_path = save_path + 'metrics.xlsx'
    df.to_excel(metrics_path, index=False)
    print(f"\nMetrics exported to {metrics_path}")
    
    pass       

if __name__ == "__main__":
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    main(config)

