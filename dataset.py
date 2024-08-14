import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transforms import transforms
import torchvision.transforms as T

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    transforms.RandomMisalignmentOverlap(shift_x=1, shift_y=1, alpha=0.9),
    transforms.RandomAddGaussianNoise(factor=0.05),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(),
])


class LIDCIDRIDataset(Dataset):
    def __init__(self, root_dir, is_train=True):
        sr = pd.read_csv('dataset/sr.csv')
        self.root_dir = root_dir
        self.transform = transform_train if is_train else transform_val
        self.sr = sr
        self.sequence = np.load(f'{root_dir}/balanced_seg_ids.npy')
        self.total_data_num = len(self.sequence)
        self.is_train = is_train

    def __len__(self):
        return self.total_data_num

    def __getitem__(self, idx):
        mask = np.load(f'{self.root_dir}/cropped_resampled_mask/{self.sequence[idx]}.npy').astype(np.float32)
        seg = np.load(f'{self.root_dir}/cropped_resampled_seg/{self.sequence[idx]}.npy').astype(np.float32)
        
        input = self.transform(mask * seg)

        row = self.sr[self.sr['seg_id'] == self.sequence[idx]]
        features = row[['diameter', 'surface_area_of_mesh', 'subtlety_score', 'lobular_pattern', 'spiculation']].values
        label = row['malignancy'].values[0]

        return input, torch.FloatTensor(features), label
