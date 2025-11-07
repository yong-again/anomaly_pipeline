from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
import cv2


class TransitorDatasets(Dataset):
    def __init__(self, data_dir, mode='train', transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.file_name = f"{mode}.csv"
        self.data = pd.read_csv(os.path.join(data_dir, self.file_name))
        self.label = self.data['label'].values if mode != 'test' else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = os.path.basename(self.data['img_path'][idx])
        image_path = os.path.join(self.data_dir, f'{self.mode}/{image_name}')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)

        if self.mode == 'test':
            return image['image']

        else:
            return image['image'], self.label[idx]

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import albumentations as A

    transform = A.Compose([
        A.Resize(256, 256),
        A.ToTensorV2()
    ])

    dataset = TransitorDatasets(data_dir='/workspace/anomaly/data/transitor', mode='test', transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    for images in dataloader:
        print(images.shape)
        break
