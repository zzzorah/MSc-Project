# Output mask .npy and ct-seg mapping csv
import pydicom as dicom
import os
import pandas as pd
import numpy as np
import math
import csv
import logging

logging.basicConfig(filename='logs/get_cropped_mask_and_seg.log', level=logging.DEBUG, format='%(asctime)s %(message)s', force=True)

root_dir = 'source/seg'
TARGET_SPACE = 50
os.makedirs(f'dataset/mask', exist_ok=True)
os.makedirs(f'dataset/seg', exist_ok=True)

def crop(origin_mask, ct_series_instance_uid, ct_sop_instance_uids, x_space, y_space, z_space): # 35~40
    origin_z, origin_y, origin_x = origin_mask.shape
    # print(origin_mask.shape)
    
    max_z, max_y, max_x = (0, 0, 0)
    min_z, min_y ,min_x = origin_mask.shape
    
    check_point = 0
    for index, slice in enumerate(origin_mask):
        if np.count_nonzero(slice) != 0:
            if check_point == 0:
                min_z = index
                check_point = 1
            else: max_z = index

            y_indices, x_indices = np.where(slice > 0)
            max_x, min_x = max([max_x, *x_indices]), min([min_x, *x_indices])
            max_y, min_y = max([max_y, *y_indices]), min([min_y, *y_indices])

    logging.debug(f'(max_x, min_x)={(max_x, min_x)}\n(max_y, min_y)={(max_y, min_y)}\n(max_z, min_z)={(max_z, min_z)}')

    added_num_x = math.ceil((TARGET_SPACE / x_space) - (max_x - min_x + 1))
    added_num_y = math.ceil((TARGET_SPACE / y_space) - (max_y - min_y + 1))
    added_num_z = math.ceil((TARGET_SPACE / z_space) - (max_z - min_z + 1))#len(origin_mask))
    logging.debug(f'target x={math.ceil((TARGET_SPACE / x_space))} y={math.ceil((TARGET_SPACE / y_space))} z={math.ceil((TARGET_SPACE / z_space))}')
    logging.debug(f'added_num_x={added_num_x}\nadded_num_y={added_num_y}\nadded_num_z={added_num_z}')

    l_added_num_x = math.floor(added_num_x / 2)
    r_added_num_x = added_num_x - l_added_num_x
    l_added_num_y = math.floor(added_num_y / 2)
    r_added_num_y = added_num_y - l_added_num_y
    l_added_num_z = math.floor(added_num_z / 2)
    r_added_num_z = added_num_z - l_added_num_z
    logging.debug(f'l_added_num_z {l_added_num_z}')
    logging.debug(f'r_added_num_z {r_added_num_z}')

    l_x, l_y, l_z = min_x - l_added_num_x, min_y - l_added_num_y, min_z - l_added_num_z
    r_x, r_y, r_z = max_x + r_added_num_x, max_y + r_added_num_y, max_z + r_added_num_z

    # Prevent getting out of boundary
    if l_x < 0:
        r_x += abs(l_x)
        l_x = 0
    if r_x > (origin_x - 1):
        l_x -= r_x - (origin_x - 1)
        r_x = origin_x - 1
    if l_y < 0:
        r_y += abs(l_y)
        l_y = 0
    if r_y > (origin_y - 1):
        l_y -= r_y - (origin_y - 1)
        r_y = origin_y - 1
    # if l_z < 0:
    #     r_z += abs(l_z)
    #     l_z = 0
    # if r_z > (origin_z - 1):
    #     l_z -= r_z - (origin_z - 1)
    #     r_z = origin_z - 1

    logging.debug(f'l_x={l_x}, r_x={r_x}')
    logging.debug(f'l_y={l_y}, r_y={r_y}')
    logging.debug(f'l_z={l_z}, r_z={r_z}')
    x_range = r_x - l_x + 1
    y_range = r_y - l_y + 1

    if added_num_z <= 0:
        cropped_mask = origin_mask[l_z : r_z + 1, l_y : r_y + 1, l_x : r_x + 1]    # crop x, y dimension
    else:
        cropped_mask = origin_mask[min_z : max_z + 1, l_y : r_y + 1, l_x : r_x + 1]    # crop x, y dimension
        logging.debug(f'concatenate {np.zeros((l_added_num_z, y_range, x_range), dtype="uint8").shape} {cropped_mask.shape} {np.zeros((r_added_num_z, y_range, x_range), dtype="uint8").shape}')
        cropped_mask = np.concatenate((np.zeros((l_added_num_z, y_range, x_range), dtype='uint8'), cropped_mask, np.zeros((r_added_num_z, y_range, x_range), dtype='uint8')), axis=0)
    
    # for ct_slice_id in ct_sop_instance_uids:
    ct_sequence = np.load(f'dataset/ct/slices/{ct_series_instance_uid}/sequence.npy')
    
    # Check segs sequence equal to slices sequence or not
    if max_z != min_z:
        index = np.where(ct_sequence == ct_sop_instance_uids[min_z])[0][0]
        # index_prev = np.where(ct_sequence == ct_sop_instance_uids[min_z])[0][0]
        index_next = np.where(ct_sequence == ct_sop_instance_uids[min_z + 1])[0][0]
        # print(index, index_next)
        if index > index_next: # seg sequence is upside down
            ct_sop_instance_uids.reverse()
    # print(f'0 {np.where(ct_sequence == ct_sop_instance_uids[min_z])}')
    # print(f'1 {np.where(ct_sequence == ct_sop_instance_uids[1])}')
    # print(f'2 {np.where(ct_sequence == ct_sop_instance_uids[2])}')
    # print(f'3 {np.where(ct_sequence == ct_sop_instance_uids[3])}')
    # print(f'4 {np.where(ct_sequence == ct_sop_instance_uids[4])}')
    # print(f'5 {np.where(ct_sequence == ct_sop_instance_uids[5])}')
    # print(f'6 {np.where(ct_sequence == ct_sop_instance_uids[max_z])}')
    
    start_index = np.where(ct_sequence == ct_sop_instance_uids[min_z])[0][0]
    end_index = np.where(ct_sequence == ct_sop_instance_uids[max_z])[0][0]
    logging.debug(f'[Before] start_index={start_index} end_index={end_index}')
    start_index = start_index - l_added_num_z
    end_index = end_index + r_added_num_z
    logging.debug(f'[Before] z range={(start_index, end_index)}')
    
    if start_index < 0:
        logging.debug(f'1???  {start_index}, {end_index}')
        end_index += abs(start_index)
        start_index = 0
        logging.debug(f'1@@@  {start_index}, {end_index}')
    if end_index > (len(ct_sequence) - 1):
        logging.debug(f'2???  {start_index}, {end_index}')
        start_index -= end_index - (len(ct_sequence) - 1)
        end_index = len(ct_sequence) - 1
        logging.debug(f'2@@@  {start_index}, {end_index}')

    logging.debug(f'[After] z range={(start_index, end_index)}')

    cropped_seg = None
    for ct_slice_id in ct_sequence[start_index: end_index + 1]:
        ct_slice = np.load(f'dataset/ct/slices/{ct_series_instance_uid}/{ct_slice_id}.npy')
        
        cropped_slice = ct_slice.reshape(1, *ct_slice.shape)[:, l_y : r_y + 1, l_x : r_x + 1]
        logging.debug(f'cropped_slice shape{cropped_slice.shape}')
        if cropped_seg is None:
            cropped_seg = cropped_slice
        else:
            cropped_seg = np.concatenate((cropped_slice, cropped_seg), axis=0)
    logging.debug(f'cropped_seg shape{cropped_seg.shape}')
    return cropped_mask, cropped_seg


seg = []
pass_seg_ids = []
for seg_series_instance_uid in os.listdir(root_dir):
    dc = dicom.read_file(f'{root_dir}/{seg_series_instance_uid}/1-1.dcm')
    mask = dc.pixel_array

    ct_series_instance_uid = dc['0008', '1115'][0]['0020', '000e'].value # ct Series Instance UID
    ct_sop_instance_uids = [item['0008', '1155'].value for item in dc['0008', '1115'][0]['0008', '114a']]
    seg.append([seg_series_instance_uid, ct_series_instance_uid, ';'.join(ct_sop_instance_uids)])

    logging.debug(f'[seg_id] {seg_series_instance_uid}')
    sr = pd.read_csv('dataset/sr.csv')
    row = sr[sr['seg_id'] == seg_series_instance_uid]
    try:
        if row['diameter'].values[0] > 40:
            logging.debug(f'[PASS] Diameter <=40 ({row["diameter"].values[0]})')
            pass_seg_ids.append([seg_series_instance_uid, 'Diameter <= 40'])
            continue
    except IndexError:
        logging.warning(f'[PASS] SR details not found')
        pass_seg_ids.append([seg_series_instance_uid, 'SR details not found'])
        continue
    
    # extend mask depth to make mask has same number of slice as ct
    new_mask, new_seg = crop(origin_mask=(mask if mask.ndim == 3 else mask.reshape(1, *mask.shape)),
                            ct_series_instance_uid=ct_series_instance_uid,
                            ct_sop_instance_uids=ct_sop_instance_uids,
                            x_space=row['horizontal_pixel_spacing'].values[0],
                            y_space=row['vertical_pixel_spacing'].values[0],
                            z_space=row['slice_thickness'].values[0])
    logging.debug(f'[New Shape] mask={new_mask.shape} seg={new_seg.shape}')
    np.save(f'dataset/mask/{seg_series_instance_uid}', new_mask)
    np.save(f'dataset/seg/{seg_series_instance_uid}', new_seg)
            
with open('dataset/seg.csv', 'w') as file:
    wr = csv.writer(file)
    wr.writerow(['seg_series_instance_uid', 'ct_series_instance_uid', 'ct_sop_instance_uids'])
    wr.writerows(seg)

with open('dataset/pass_seg_ids.csv', 'w') as file:
    wr = csv.writer(file)
    wr.writerow(['seg_series_instance_uid', 'reason'])
    wr.writerows(pass_seg_ids)

# print(len(os.listdir('dataset/mask')))