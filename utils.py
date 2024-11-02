from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms

class DenoisingDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.image_paths = self._load_image_paths()

    def _load_image_paths(self):
        image_paths = []
        for item in os.listdir(self.root_dir):
            item_path = os.path.join(self.root_dir, item)
            if self.mode == 'train':
                data_path = os.path.join(item_path, 'Train')
            else:
                data_path = os.path.join(item_path, 'Val')
            clean_image_path = os.path.join(data_path, 'GT_clean_image')
            noisy_image_path = os.path.join(data_path, 'Degraded_image')
            defect_mask_path = os.path.join(data_path, 'Defect_mask')
            for defect_type in os.listdir(clean_image_path):
                for file_name in os.listdir(os.path.join(clean_image_path, defect_type)):
                    if file_name.endswith('.png'):
                        gt_clean_image = os.path.join(clean_image_path, defect_type, file_name)
                        noisy_image = os.path.join(noisy_image_path, defect_type, file_name)
                        file_name = file_name.replace('.png', '_mask.png')
                        defect_mask = os.path.join(defect_mask_path, defect_type, file_name)
                        image_paths.append((gt_clean_image, noisy_image, defect_mask))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        gt_clean_image_path, noisy_image_path, defect_mask_path = self.image_paths[idx]

        gt_clean_image = Image.open(gt_clean_image_path).convert("RGB")
        noisy_image = Image.open(noisy_image_path).convert("RGB")
        defect_mask = Image.open(defect_mask_path).convert("L")

        if self.transform:
            gt_clean_image = self.transform(gt_clean_image)
            noisy_image = self.transform(noisy_image)
            defect_mask = self.transform(defect_mask)

        return gt_clean_image, noisy_image, defect_mask