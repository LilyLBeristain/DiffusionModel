import os
from pathlib import Path
from itertools import chain
from PIL import Image
import torch
import shutil
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import kagglehub

class DataModule(Dataset):
    def __init__(self, root='data', split='train', batch_size=32, num_workers=4, image_resolution=64, max_images=-1, label_offset=1):
        self.root = Path(root) / split
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_resolution = image_resolution
        self.max_images = max_images
        self.label_offset = label_offset

        os.makedirs(self.root, exist_ok=True)
        
        if not any(self.root.iterdir()):
            self._download_dataset()
        
        self.transform = transforms.Compose([
            transforms.Resize((self.image_resolution, self.image_resolution)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.fnames, self.labels = self._load_images()
        self.num_classes = len(set(self.labels))

    def _download_dataset(self):
        download_path = kagglehub.dataset_download("andrewmvd/animal-faces")
        target_path = self.root  # Ensure dataset is stored in `data/dataset`
        # check if there is any file in target path
        if not any(target_path.iterdir()):
            download_path = os.path.join(download_path, 'afhq', self.split)
            for item in os.listdir(download_path):
                src = os.path.join(download_path, item)
                dst = target_path / item
                print(f'Copying {item} to {target_path}')
                shutil.copytree(src, dst, dirs_exist_ok=True)  # <--- cambio aquí
            # No intentar borrar la carpeta de Kaggle porque es solo lectura
        
    def _load_images(self):
        categories = sorted(self.root.iterdir())
        fnames, labels = [], []
        for idx, cat in enumerate(categories):
            cat_fnames = sorted(self._list_images(cat))[:self.max_images if self.max_images > 0 else None]
            fnames.extend(cat_fnames)
            labels.extend([idx + self.label_offset] * len(cat_fnames))
        return fnames, labels
    
    def _list_images(self, path):
        return chain(*(path.rglob(f'*.{ext}') for ext in ['png', 'jpg', 'jpeg', 'JPG']))
    
    def __getitem__(self, idx):
        img = Image.open(self.fnames[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]
    
    def __len__(self):
        return len(self.labels)
    
    def dataloader(self, shuffle=True, drop_last=True):
        return DataLoader(self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle, drop_last=drop_last)

def tensor_to_pil_image(tensor):
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1).detach().cpu().permute(0, 2, 3, 1).numpy()
    return [Image.fromarray((img * 255).astype('uint8')) for img in tensor]

def get_data_iterator(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

def evaluation_dataset(data_root='data', batch_size=32, num_workers=4, image_resolution=64):
    dataset = DataModule(data_root, 'val', batch_size, num_workers, image_resolution)
    eval_dir = dataset.root.parent / 'eval'
    eval_dir.mkdir(exist_ok=True)
    
    for path in dataset.fnames:
        img = Image.open(path).resize((64, 64))
        img.save(eval_dir / path.name)
        print(f'Processed {path.name}')
    
    print(f'Constructed eval directory at {eval_dir}')
    
if __name__ == "__main__":
    ds_module = DataModule("./data", "train", 32, num_workers=4, image_resolution=64)
    evaluation_dataset()
    