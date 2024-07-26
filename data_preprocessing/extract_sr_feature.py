# Extract features from SR documents to csv
import pydicom as dicom
import os
import csv
import logging

logging.basicConfig(filename='logs/extract_sr_feature.log', level=logging.DEBUG, force=True)

root_dir = 'source/sr'
file_map = {}

sr_field = ['id', 'seg_id', 'ct_id', 'horizontal_pixel_spacing', 'vertical_pixel_spacing', 'slice_thickness', 'diameter', 'surface_area_of_mesh', 'subtlety_score', 'lobular_pattern', 'spiculation', 'malignancy']
sr = []
err = []
for id in os.listdir(root_dir):
    print(f'{root_dir}/{id}/1-1.dcm')
    dc = dicom.read_file(f'{root_dir}/{id}/1-1.dcm')
    # id = dc['SeriesInstanceUID'].value
    try:
        seg_id = dc['CurrentRequestedProcedureEvidenceSequence'][0]['ReferencedSeriesSequence'][0]['SeriesInstanceUID'].value
        ct_id = dc['CurrentRequestedProcedureEvidenceSequence'][0]['ReferencedSeriesSequence'][1]['SeriesInstanceUID'].value

        horizontal_pixel_spacing = None
        vertical_pixel_spacing = None
        slice_thickness = None
        for item in dc['0040', 'a730'][4]['0040', 'a730'][0]['0040', 'a730']:
            if ['0040', 'a043'] in item:
                if item['0040', 'a043'][0]['0008', '0104'].value == 'Horizontal Pixel Spacing':
                    horizontal_pixel_spacing = item['0040', 'a300'][0]['0040', 'a30a'].value
                elif item['0040', 'a043'][0]['0008', '0104'].value == 'Vertical Pixel Spacing':
                    vertical_pixel_spacing = item['0040', 'a300'][0]['0040', 'a30a'].value
                elif item['0040', 'a043'][0]['0008', '0104'].value == 'Slice Thickness':
                    slice_thickness = item['0040', 'a300'][0]['0040', 'a30a'].value

        diameter = None
        surface_area_of_mesh = None
        subtlety_score = None
        # margin = None
        lobular_pattern = None
        spiculation = None
        malignancy = None
        for item in dc['0040', 'a730'][5]['0040', 'a730'][0]['0040', 'a730']:
            if item['0040', 'a043'][0]['0008', '0104'].value == 'Diameter':
                diameter = item['0040', 'a300'][0]['0040', 'a30a'].value
            elif item['0040', 'a043'][0]['0008', '0104'].value == 'Surface area of mesh':
                surface_area_of_mesh = item['0040', 'a300'][0]['0040', 'a30a'].value
            elif item['0040', 'a043'][0]['0008', '0104'].value == 'Subtlety score':
                subtlety_score = item['0040', 'a168'][0]['0008', '0104'].value[0]
            elif item['0040', 'a043'][0]['0008', '0104'].value == 'Lobular Pattern':
                lobular_pattern = item['0040', 'a168'][0]['0008', '0104'].value[0]
            elif item['0040', 'a043'][0]['0008', '0104'].value == 'Spiculation':
                spiculation = item['0040', 'a168'][0]['0008', '0104'].value[0]
            elif item['0040', 'a043'][0]['0008', '0104'].value == 'Malignancy':
                malignancy = item['0040', 'a168'][0]['0008', '0104'].value[0]
    except IndexError as e:
        err.append([id])
        logging.error(f'[IndexError] id={id}\n{e}')

    sr.append([id, seg_id, ct_id,horizontal_pixel_spacing, vertical_pixel_spacing, slice_thickness, diameter, surface_area_of_mesh, subtlety_score, lobular_pattern, spiculation, malignancy])
logging.info("[Finish] read all files")

with open('dataset/sr.csv', 'w') as file:
    wr = csv.writer(file)
    wr.writerow(sr_field)
    wr.writerows(sr)
    logging.info("[Finish] write sr.csv")


with open('dataset/sr_err.csv', 'w') as file:
    wr = csv.writer(file)
    wr.writerows(err)
    logging.info("[Finish] write sr_err.csv")

