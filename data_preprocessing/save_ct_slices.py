# Save slices to .npy divided by CT Series Instance UID
import pydicom as dicom
import os
import errno
import numpy as np
# import logging

# logging.basicConfig(filename='save_ct_slices.log', level=logging.DEBUG)

root_dir = 'source/ct'

for ct_series_instance_uid in os.listdir(root_dir):
    instance_sequence = []
    for filename in sorted(os.listdir(f'{root_dir}/{ct_series_instance_uid}')):
        if filename == 'LICENSE': continue

        dc = dicom.read_file(f'{root_dir}/{ct_series_instance_uid}/{filename}')
        instance_sequence.append(dc["SOPInstanceUID"].value)
        
        os.makedirs(f'dataset/ct/slices/{dc["SeriesInstanceUID"].value}', exist_ok=True)
        np.save(f'dataset/ct/slices/{dc["SeriesInstanceUID"].value}/{dc["SOPInstanceUID"].value}', dc.pixel_array)
    np.save(f'dataset/ct/slices/{dc["SeriesInstanceUID"].value}/sequence', instance_sequence)