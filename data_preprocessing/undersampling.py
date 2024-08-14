import pandas as pd
import numpy as np
import random

EACH_LABEL_NUM = 600

df = pd.read_csv('dataset/sr.csv')
ids = np.load('dataset/seg_ids.npy').tolist()
rows = df[(df['seg_id'].isin(ids)) & (df['diameter']<=35)]
# print(rows['malignancy'].value_counts().sort_index())

classify_ids = [[], [], [], [], []]
for _, row in rows.iterrows():
    index = row['malignancy'] - 1
    classify_ids[index].append(row['seg_id'])

new_ids = []
for index in range(5):
    new_ids += random.sample(classify_ids[index], EACH_LABEL_NUM)
random.shuffle(new_ids)
np.save('dataset/balanced_seg_ids', new_ids)

test_ids = set(ids) - set(new_ids)
np.save('dataset/test_ids', test_ids)

print(f'train: {len(new_ids)}, test: {len(test_ids)}, total: {len(ids)}')
