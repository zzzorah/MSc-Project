import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from package import transforms

transform_train = transforms.Compose([
    # transforms.RandomScale(range(28, 38)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomYFlip(),
    transforms.RandomZFlip(),
    transforms.ZeroOut(4),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),  # need to cal mean and std, revise norm func
])


class LIDCIDRIDataset(Dataset):
    def __init__(self, root_dir, transform=transform_train):
        sr = pd.read_csv('dataset/sr.csv')

        self.root_dir = root_dir
        self.transform = transform
        self.sr = sr
        self.sequence = np.load(f'{root_dir}/seg_ids.npy')
        self.total_data_num = len(self.sequence)

    def __len__(self):
        return self.total_data_num

    def __getitem__(self, idx):
        mask = np.load(f'{self.root_dir}/resampled_mask/{self.sequence[idx]}.npy').astype(np.float32)
        seg = np.load(f'{self.root_dir}/resampled_seg/{self.sequence[idx]}.npy').astype(np.float32)
        
        # mask = np.load(f'{self.root_dir}/mask/{self.sequence[idx]}.npy')
        # seg = np.load(f'{self.root_dir}/seg/{self.sequence[idx]}.npy')
        input = self.transform(mask * seg)

        row = self.sr[self.sr['seg_id'] == self.sequence[idx]]
        features = row[['diameter', 'surface_area_of_mesh', 'subtlety_score', 'lobular_pattern', 'spiculation']].values
        label = row['malignancy'].values[0]

        return input, label

# ids = np.load('dataset/seg_ids.npy')
# max_hu = 0
# min_hu = 3000
# count = 0
# for id in ids:
#     count += 1
#     max_hu = max(np.max(np.load(f'dataset/cropped_resampled_seg/{id}.npy')), max_hu)
#     min_hu = min(np.min(np.load(f'dataset/cropped_resampled_seg/{id}.npy')), min_hu)
# print(f'(min_hu, max_hu) = {(min_hu, max_hu)}, count = {count}')

# nomalization of hu value
# def transform(pixels):
#     new_pixels = (pixels - min_hu) / (max_hu - min_hu)
#     new_pixels = new_pixels.astype(np.float32)
#     # new_pixels = new_pixels.reshape(1, *new_pixels.shape)
#     # print(f'pixel shape = {new_pixels.shape}')
    
#     return new_pixels # new_pixels
