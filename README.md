# MSc Project
This project is an implementation of a dissertation, which modifies NASLung[2][3][4].

### Model Architecture
![architecture](https://github.com/user-attachments/assets/01fa6a5a-73bf-4835-9012-b45e8c82b41e)


## Directory
|_ _ data_preprocessing<br>
|&emsp;&emsp;&emsp;|_ _ extract_sr_feature.py<br>
|&emsp;&emsp;&emsp;|_ _ get_cropped_mask_and_seg.py<br>
|&emsp;&emsp;&emsp;|_ _ resample.py<br>
|&emsp;&emsp;&emsp;|_ _ save_ct_slices.py<br>
|&emsp;&emsp;&emsp;|_ _ undersampling.py<br>
|_ _ dataset<br>
|_ _ logs<br>
|&emsp;&emsp;&emsp;|_ _ logging_config.py<br>
|_ _ model<br>
|&emsp;&emsp;&emsp;|_ _ angle_linear.py<br>
|&emsp;&emsp;&emsp;|_ _ ca_block.py<br>
|&emsp;&emsp;&emsp;|_ _ cnn.py<br>
|&emsp;&emsp;&emsp;|_ _ conv3d_layer.py<br>
|&emsp;&emsp;&emsp;|_ _ drdb.py<br>
|&emsp;&emsp;&emsp;|_ _ fusion_model.py<br>
|&emsp;&emsp;&emsp;|_ _ mlp.py<br>
|&emsp;&emsp;&emsp;|_ _ res_cbam_layer.py<br>
|&emsp;&emsp;&emsp;|_ _ utils.py<br>
|_ _ results<br>
|&emsp;&emsp;&emsp;|_ _ models<br>
|_ _ source<br>
|_ _ transforms<br>
|&emsp;&emsp;&emsp;|_ _ transforms.py<br>
|_ _ dataset.py<br>
|_ _ train_model.py<br>

## Dataset
[The Lung Image Database Consortium image collection (LIDC-IDRI)](https://www.cancerimagingarchive.net/collection/lidc-idri/) is used as the dataset. All the data is in DICOM format.
### Step
1. Download the dataset from the link mentioned above and access it through the [NBIA Data Retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images)
*Suggestion: Download CT images, segmentation (SEG) files, and SR documents separately to simplify the data preprocessing process.*
2. Save the CT images, SEG files, and SR documents in source/ct, source/seg, and source/sr directories respectively.
3. Run the preprocessing codes.
    ```sh
    python3 save_ct_slices.py  # Save each slice as a .npy in dataset/ct
    python3 extract_sr_feature.py  # Save all features into dataset/sr.csv
    python3 get_cropped_mask_and_seg.py  # Cropped images with fewwer pixels as .npy 
    python3 resample.py  # Do resampling and crop into expected size
    python3 undersampling.py  # Deal with data unbalanced problem
    ```

## Train Model
To train the model, run:
```
python3 train.py
```
All logs are saved in the logs/ directory as *.log files. Models from each fold will be saved in the results/models directory, and validation results from each epoch will be saved as *.pt files in the results/ directory.

## Reference
1. Armato III, S. G., McLennan, G., Bidaut, L., McNitt-Gray, M. F., Meyer, C. R., Reeves, A. P., Zhao, B., Aberle, D. R., Henschke, C. I., Hoffman, E. A., Kazerooni, E. A., MacMahon, H., Van Beek, E. J. R., Yankelevitz, D., Biancardi, A. M., Bland, P. H., Brown, M. S., Engelmann, R. M., Laderach, G. E., Max, D., Pais, R. C. , Qing, D. P. Y. , Roberts, R. Y., Smith, A. R., Starkey, A., Batra, P., Caligiuri, P., Farooqi, A., Gladish, G. W., Jude, C. M., Munden, R. F., Petkovska, I., Quint, L. E., Schwartz, L. H., Sundaram, B., Dodd, L. E., Fenimore, C., Gur, D., Petrick, N., Freymann, J., Kirby, J., Hughes, B., Casteele, A. V., Gupte, S., Sallam, M., Heath, M. D., Kuhn, M. H., Dharaiya, E., Burns, R., Fryd, D. S., Salganicoff, M., Anand, V., Shreter, U., Vastagh, S., Croft, B. Y., Clarke, L. P. (2015). Data From LIDC-IDRI [[Data set](https://www.cancerimagingarchive.net/collection/lidc-idri/)]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX
2. Hanliang Jiang, Fuhao Shen, Fei Gao, Weidong Han, Learning efficient, explainable and discriminative representations for pulmonary nodules classification, Pattern Recognition, Volume 113, 2021, 107825, ISSN 0031-3203, https://doi.org/10.1016/j.patcog.2021.107825.
3. Fei-aiart, (2021), NAS-Lung, GitHub repository, https://github.com/fei-aiart/NAS-Lung
4. Lin, CY., Guo, SM., Lien, JJ.J. et al. Combined model integrating deep learning, radiomics, and clinical data to classify lung nodules at chest CT. Radiol med 129, 56–69 (2024). https://doi.org/10.1007/s11547-023-01730-6
