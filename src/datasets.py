from torch.utils.data import Dataset
import os
from pathlib import Path
import pandas as pd
from PIL import Image


class TransitorDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.data = []
        self.mode = mode
        self.file_name = f'{mode}.csv'

        # Load data from CSV files
        for csv_file in self.root_dir.glob(self.file_name):
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                image_path = self.root_dir / row['img_path']
                if mode == 'test':
                    label = None
                else:
                    label = row['label']
                self.data.append((image_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.mode == 'test':
            return image
        else:
            return image, label

if __name__ == '__main__':
    # Example
    from torchvision import transforms
    from torch.utils.data import DataLoader
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = TransitorDataset(root_dir='../data/transitor/', transform=transform, mode='train')
    print(f'Dataset size: {len(dataset)}')

    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for images, labels in train_dataloader:
        print(f'Batch size: {images.size(0)}')
        break

    test_dataset = TransitorDataset(root_dir='../data/transitor/', transform=transform, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    for images in test_dataloader:
        print(f'Test Batch size: {images.size(0)}')
        break