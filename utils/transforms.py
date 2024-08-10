from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps

try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
from torch.autograd import Variable

# torch.cuda.set_device(0)


def resample3d(inp, inp_space, out_space=(1, 1, 1)):
    # Infer new shape
    # inp = torch.from_numpy(inp)
    # inp=torch.FloatTensor(inp)
    # inp=Variable(inp)
    inp = inp.cuda()
    out = resample1d(inp, inp_space[2], out_space[2]).permute(0, 2, 1)
    out = resample1d(out, inp_space[1], out_space[1]).permute(2, 1, 0)
    out = resample1d(out, inp_space[0], out_space[0]).permute(2, 0, 1)
    return out


def resample1d(inp, inp_space, out_space=1):
    # Output shape
    print(inp.size(), inp_space, out_space)
    out_shape = list(np.int64(inp.size()[:-1])) + [
        int(np.floor(inp.size()[-1] * inp_space / out_space))]  # Optional for if we expect a float_tensor
    out_shape = [int(item) for item in out_shape]
    # Get output coordinates, deltas, and t (chord distances)
    # torch.cuda.set_device(inp.get_device())
    # Output coordinates in real space
    coords = torch.cuda.HalfTensor(range(out_shape[-1])) * out_space
    delta = coords.fmod(inp_space).div(inp_space).repeat(out_shape[0], out_shape[1], 1)
    t = torch.cuda.HalfTensor(4, out_shape[0], out_shape[1], out_shape[2]).zero_()
    t[0] = 1
    t[1] = delta
    t[2] = delta ** 2
    t[3] = delta ** 3
    # Nearest neighbours indices
    nn = coords.div(inp_space).floor().long()
    # Stack the nearest neighbors into P, the Points Array
    P = torch.cuda.HalfTensor(4, out_shape[0], out_shape[1], out_shape[2]).zero_()
    for i in range(-1, 3):
        P[i + 1] = inp.index_select(2, torch.clamp(nn + i, 0, inp.size()[-1] - 1))
        # Take catmull-rom  spline interpolation:
    return 0.5 * t.mul(torch.cuda.HalfTensor([[0, 2, 0, 0],
                                              [-1, 0, 1, 0],
                                              [2, -5, 4, -1],
                                              [-1, 3, -3, 1]]).mm(P.view(4, -1)) \
                       .view(4,
                             out_shape[0],
                             out_shape[1],
                             out_shape[2])) \
        .sum(0) \
        .squeeze()


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            # print(t)
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


from scipy.ndimage.interpolation import zoom


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
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            pic = np.expand_dims(pic, -1)
            img = torch.from_numpy(pic.transpose((3, 0, 1, 2)))
            # backward compatibility
            return img.float()  # .div(255)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()  # .div(255)
        else:
            return img



class RandomHorizontalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return np.array(img[:, :, ::-1])  # .transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomZFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return np.array(img[::-1, :, :])
        return img


class RandomYFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return np.array(img[:, ::-1, :])
        return img

