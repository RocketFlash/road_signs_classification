import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

def my_split(values, n_val_samples, min_n_samples=4):
    X_val = np.array([])
    if len(values) >= min_n_samples:
        X_val = values[:n_val_samples]
        X_train = values[n_val_samples:]
    else:
        X_train = values

    return X_train, X_val

DATASET_PATH = '/home/rauf/datasets/RTSD/'
SUBDIRS = ['rtsd-r1/', 'rtsd-r3/']
DATA_SUBSETS = ['train', 'test']
SPLIT_NAME = 'split_simple_1'
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

images_paths = {}
val_ratio = 0.2
max_n_of_files_class = 100000000000
min_n_of_files_class = 1

N_SAMPLES_VAL = 2
MIN_N_SAMPLES_VAL = 4

numbers_to_classes_file = 'numbers_to_classes.csv'

save_path = os.path.join(DATASET_PATH, 'splits', f'{SPLIT_NAME}/')
os.makedirs(os.path.join(save_path), exist_ok=True)


for SUBDIR_PATH in SUBDIRS:
    train_images = []
    test_images = []
    full_dir_path = f'{DATASET_PATH}{SUBDIR_PATH}'
    with open(f'{full_dir_path}{numbers_to_classes_file}', mode='r') as infile:
        reader = csv.reader(infile)
        numbers_to_classes_mapper = {rows[0]: rows[1] for rows in reader}

    for d in DATA_SUBSETS:
        labels_file = f'{full_dir_path}gt_{d}.csv'
        with open(labels_file, "r") as f:
            reader = csv.reader(f, delimiter=",")
            for i, line in enumerate(reader):
                if i == 0:
                    continue
                cl = numbers_to_classes_mapper[line[1]]
                if cl not in images_paths:
                    images_paths[cl] = []
                images_paths[cl].append(os.path.join(SUBDIR_PATH, d, line[0]))

print(f'Total N of classes: {len(images_paths)}')
count = 0
train_paths, train_classes = [], []
val_paths, val_classes = [], []

for key, value in images_paths.items():
    values_np = np.array(value)

    if values_np.shape[0] < min_n_of_files_class:
        print('Class {} was skipped'.format(key))
        continue

    if values_np.shape[0] >= max_n_of_files_class:
        values_np = np.random.choice(values_np, max_n_of_files_class, replace=False)

    X_train, X_val = my_split(values_np, N_SAMPLES_VAL, min_n_samples=MIN_N_SAMPLES_VAL)
    train_paths += list(X_train)
    train_classes += [key] * len(X_train)
    val_paths += list(X_val)
    val_classes += [key] * len(X_val)

df_train = pd.DataFrame(list(zip(train_paths, train_classes)), columns = ['file_path','class_name'])
df_val = pd.DataFrame(list(zip(val_paths, val_classes)), columns = ['file_path','class_name'])
df_train.to_csv(f'{save_path}train.csv', index=False)
df_val.to_csv(f'{save_path}val.csv', index=False)