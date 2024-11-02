import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import DenoisingDataset
from model.sadnet import SADNET
from losses import MSELoss, PSNRLoss, L1Loss
from metrics import calculate_psnr, calculate_ssim
import yaml

def main(config):
    transform = transforms.Compose([
    transforms.Resize((config['train']['input_size'], config['train']['input_size'])),
    transforms.ToTensor(),
    ])

    train_dataset = DenoisingDataset(root_dir=config['train']['data_path'], mode='train', transform=transform)
    val_dataset = DenoisingDataset(root_dir=config['val']['data_path'], mode='val', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=2,pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['val']['batch_size'], shuffle=False, num_workers=2,pin_memory=True, drop_last=False)
    
    denoising_model = SADNET(input_channel=3,output_channel=3)
    
    if config['train']['model_checkpoint']:
        denoising_model.load_state_dict(torch.load(config['train']['model_checkpoint']))
    
    optimizer = torch.optim.Adam(denoising_model.parameters(), lr=config['train']['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['train']['T_max'], eta_min=config['train']['eta_min'])
    
    mseLoss = MSELoss(reduction=config['train']['loss_reduction'])
    l1Loss = L1Loss(reduction=config['train']['loss_reduction'])
    psnrLoss = PSNRLoss(reduction=config['train']['loss_reduction'])
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    denoising_model.to(device)
    #Hyperparameters
    alpha, beta = config['train']['alpha'], config['train']['beta']
    c = config['train']['loss_weights']
    # Number of epochs
    best_loss = float('inf')
    num_epochs = config['train']['num_epochs']

    # Training loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Training phase
        denoising_model.train()
        train_loss = 0.0
        for _, clean_images, degraded_images, masks in tqdm(train_loader):
            clean_images = clean_images.to(device)
            degraded_images = degraded_images.to(device)
            masks = masks.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = denoising_model(degraded_images)

            # Compute loss
            weights = alpha + beta * masks
            loss1 = mseLoss(outputs, clean_images, weights)
            loss2 = l1Loss(outputs, clean_images, weights)
            loss3 = psnrLoss(outputs, clean_images)
            loss = c[0]*loss1 + c[1]*loss2 + c[2]*loss3

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * clean_images.size(0)

        # Calculate average training loss
        epoch_train_loss = train_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1} - Training Loss: {epoch_train_loss:.4f}')

        # Validation phase
        denoising_model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        with torch.no_grad():
            for _, clean_images, degraded_images, masks in tqdm(val_loader):
                clean_images = clean_images.to(device)
                degraded_images = degraded_images.to(device)
                masks = masks.to(device)
                outputs = denoising_model(degraded_images)

                # Compute loss
                weights = alpha + beta * masks
                loss1 = mseLoss(outputs, clean_images, weights)
                loss2 = l1Loss(outputs, clean_images, weights)
                loss3 = psnrLoss(outputs, clean_images)
                loss = c[0]*loss1 + c[1]*loss2 + c[2]*loss3
                val_loss += loss.item() * clean_images.size(0)

                # Compute PSNR
                outputs = torch.clamp(outputs, 0, 1)
                batch_psnr = calculate_psnr(clean_images, outputs, max_pixel_value=1.0)
                batch_ssim = calculate_ssim(clean_images, outputs, max_pixel_value=1.0)
                val_psnr += batch_psnr * clean_images.size(0)
                val_ssim += batch_ssim * clean_images.size(0)


        # Calculate average validation loss and accuracy
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_psnr /= len(val_loader.dataset)
        val_ssim /= len(val_loader.dataset)


        print(f'Validation Loss: {epoch_val_loss:.4f}')
        print(f'Validation PSNR: {val_psnr:.4f}')
        print(f'Validation SSIM: {val_ssim:.4f}')

        # Step the scheduler
        scheduler.step()

        # Save the model if it has the best validation loss
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            torch.save(denoising_model.state_dict(), config['train']['model_save_path'])
            print(f"Best model saved at epoch {epoch + 1} with validation loss: {epoch_val_loss:.4f}")
     


if __name__ == '__main__':
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    main(config)


