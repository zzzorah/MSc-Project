import random
import torch
import torch.nn.functional as F


class RandomMisalignmentOverlap(object):
    def __init__(self, shift_x=1, shift_y=1, alpha=0.65, probability=0.5):
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.alpha = alpha
        self.probability = probability

    def __call__(self, origin_image):
        if random.random() < self.probability:
            theta = torch.tensor([
                [1, 0, 0, 0.5],
                [0, 1, 0, 0.5],
                [0, 0, 1, 0.5]
            ])
            theta = theta.unsqueeze(0)
            origin_image = origin_image.unsqueeze(0)

            affine_grid = F.affine_grid(theta, origin_image.size(), align_corners=False)
            transformed_image = F.grid_sample(origin_image, affine_grid, align_corners=False)
            blended_image = origin_image * self.alpha + transformed_image * (1 - self.alpha)

            blended_image = blended_image.squeeze(0)
            return blended_image
        else: return origin_image
