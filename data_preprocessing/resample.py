# Resample
import pandas as pd
import numpy as np
import os
from scipy.ndimage import zoom
import logging

logging.basicConfig(filename='logs/resample.log', level=logging.DEBUG, format='%(asctime)s %(message)s', force=True)

def resample(mask, seg, origin_spacing, min_spacing, origin_shape):
    resize_factor = np.array(origin_spacing) / np.array(min_spacing)
    logging.debug(resize_factor)
    new_shape = np.round(origin_shape * resize_factor)
    logging.debug(f'new shape = {new_shape}')
    resize_factor = new_shape / origin_shape
    resampled_mask = zoom(mask, resize_factor, mode='nearest')
    resampled_seg = zoom(seg, resize_factor, mode='nearest')
    logging.debug(f'original shape = {origin_shape}, resampled mask shape = {resampled_mask.shape}, resampled seg shape = {resampled_seg.shape}')
    return resampled_mask, resampled_seg

def crop(mask, seg, spacing, target_space=32):
    TARGET_SPACE = target_space
    max_z, max_y, max_x = (0, 0, 0)
    min_z, min_y ,min_x = mask.shape
    origin_shape = mask.shape
    
    z_indices, y_indices, x_indices = np.where(mask > 0)
    max_x, min_x = max([max_x, *x_indices]), min([min_x, *x_indices])
    max_y, min_y = max([max_y, *y_indices]), min([min_y, *y_indices])
    max_z, min_z = max([max_z, *z_indices]), min([min_z, *z_indices])
    centre_index = np.ceil(np.array([(max_z + min_z) / 2, (max_y + min_y) / 2, (max_x + min_x) / 2]))
    logging.debug(f'(max_z, min_z)={(max_z, min_z)} (max_x, min_x)={(max_x, min_x)}, (max_y, min_y)={(max_y, min_y)}, ')
    logging.debug(f'centre_index = {centre_index}')

    target_index_num = np.ceil(TARGET_SPACE / spacing)
    logging.debug(f'target_indices = {target_index_num}')

    l_index = centre_index - np.floor((target_index_num - 1) / 2).astype(int)
    r_index = centre_index + np.ceil((target_index_num - 1) / 2).astype(int)
    logging.debug(f'[Before] l_index = {l_index}, r_index = {r_index}')

    for i, value in enumerate(l_index):
        if value < 0:
            r_index[i] += abs(value)
            l_index[i] = 0
    for i, value in enumerate(r_index):
        if value > (origin_shape[i] - 1):
            l_index[i] -= value - (origin_shape[i] - 1)
            r_index[i] = origin_shape[i] - 1
    
    l_index = l_index.astype(int)
    r_index = r_index.astype(int)
    
    logging.debug(f'[After] l_index = {l_index}, r_index = {r_index}')

    cropped_mask = mask[l_index[0] : r_index[0] + 1, l_index[1] : r_index[1] + 1, l_index[2] : r_index[2] + 1]    # crop x, y dimension
    cropped_seg = seg[l_index[0] : r_index[0] + 1, l_index[1] : r_index[1] + 1, l_index[2] : r_index[2] + 1]    # crop x, y dimension

    return cropped_mask, cropped_seg


sr = pd.read_csv('dataset/sr.csv')
min_x_spacing = sr['horizontal_pixel_spacing'].min()
min_y_spacing = sr['vertical_pixel_spacing'].min()
min_z_spacing = sr['slice_thickness'].min()
logging.debug(f'min_x_spacing={min_x_spacing}, min_y_spacing={min_y_spacing}, min_z_spacing={min_z_spacing}')

seg_ids = []
for filename in os.listdir('dataset/seg'):
    if '.npy' not in filename:  continue
    seg_series_instance_uid = filename.split('.npy')[0]
    logging.debug(f'seg_series_instance_uid = {seg_series_instance_uid}')
    
    mask = np.load(f'dataset/mask/{filename}')
    seg = np.load(f'dataset/seg/{filename}')

    row = sr[sr['seg_id'] == seg_series_instance_uid]
    resampled_mask, resampled_seg = resample(mask=mask,
                                             seg=seg,
                                             origin_spacing=[row['slice_thickness'].values[0], row['vertical_pixel_spacing'].values[0], row['horizontal_pixel_spacing'].values[0]],
                                             min_spacing=[min_z_spacing, min_y_spacing, min_x_spacing],
                                             origin_shape=mask.shape)
    # cropped_mask, cropped_seg = crop(mask=resampled_mask, seg=resampled_seg, spacing=np.array([min_z_spacing, min_y_spacing, min_x_spacing]))

    # os.makedirs('dataset/cropped_resampled_mask', exist_ok=True)
    # os.makedirs('dataset/cropped_resampled_seg', exist_ok=True)
    # np.save(f'dataset/cropped_resampled_mask/{seg_series_instance_uid}', cropped_mask)
    # np.save(f'dataset/cropped_resampled_seg/{seg_series_instance_uid}', cropped_seg)
    os.makedirs('dataset/resampled_mask', exist_ok=True)
    os.makedirs('dataset/resampled_seg', exist_ok=True)
    np.save(f'dataset/resampled_mask/{seg_series_instance_uid}', resampled_mask)
    np.save(f'dataset/resampled_seg/{seg_series_instance_uid}', resampled_seg)
    seg_ids.append(seg_series_instance_uid)
np.save(f'dataset/seg_ids', seg_ids)
logging.debug('Run all files.')
