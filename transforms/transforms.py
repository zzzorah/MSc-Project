import torch
import random
import numpy as np
import torch.nn.functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Normalize(object):
    def __init__(self):
        ids = np.load('dataset/balanced_seg_ids.npy')
        max_hu = 0
        min_hu = 3000
        count = 0

        for id in ids:
            seg = np.load(f'dataset/cropped_resampled_seg/{id}.npy')
            count += 1
            max_hu = max(np.max(seg), max_hu)
            min_hu = min(np.min(seg), min_hu)
        print(f'(min_hu, max_hu) = {(min_hu, max_hu)}, count = {count}')

        self.max_hu = max_hu
        self.min_hu = min_hu
        self.count = count

    def __call__(self, pixels):
        new_pixels = (pixels - self.min_hu) / (self.max_hu - self.min_hu)
        return new_pixels


class ZeroOut(object):
    def __init__(self, size):
        self.size = int(size)

    def __call__(self, img):
        w, h, d = img.shape
        x1 = random.randint(0, w - self.size)
        y1 = random.randint(0, h - self.size)
        z1 = random.randint(0, d - self.size)

        img1 = np.array(img)
        img1[x1:x1 + self.size, y1:y1 + self.size, z1:z1 + self.size] = np.array(
            np.zeros((self.size, self.size, self.size)))
        return np.array(img1)


class ToTensor(object):
    def __call__(self, input):
        return torch.from_numpy(input.astype(np.float32)).unsqueeze(0)


class RandomZFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return np.array(img[::-1, :, :])
        return img


class RandomMisalignmentOverlap(object):
    def __init__(self, shift_x=1, shift_y=1, alpha=0.65, probability=0.5):
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.alpha = alpha
        self.probability = probability

    def __call__(self, origin_image):
        if random.random() < self.probability:
            theta = torch.tensor([
                [1, 0, 0, self.shift_x],
                [0, 1, 0, self.shift_y],
                [0, 0, 1, 0]
            ], dtype=torch.float32)
            theta = theta.unsqueeze(0)
            origin_image = origin_image.unsqueeze(0)

            affine_grid = F.affine_grid(theta, origin_image.size(), align_corners=False)
            transformed_image = F.grid_sample(origin_image, affine_grid, align_corners=False)
            blended_image = origin_image * self.alpha + transformed_image * (1 - self.alpha)

            blended_image = blended_image.squeeze(0)
            return blended_image
        else: return origin_image


class RandomAddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0, probability=0.5, factor=0.3):
        self.mean = mean
        self.std = std
        self.probability = probability
        self.factor = factor

    def __call__(self, origin_image):
        if random.random() < self.probability:
            transformed_image = origin_image + torch.randn_like(origin_image) * self.factor
            transformed_image = torch.clip(transformed_image, 0., 1.)
            return transformed_image
        return origin_image