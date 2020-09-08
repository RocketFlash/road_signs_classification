import pandas as pd
import cv2
import os
from torch.utils.data import Dataset
from .dataset import generate_test_df

class RTSDDataset(Dataset):
    def __init__(self, data_dir, df_path=None, mapper=None, transform=None, img_ext='.png'):
        super().__init__()
        self.df = pd.read_csv(df_path) if df_path is not None else generate_test_df(data_dir, column_name='file_path', 
                                                                                              img_ext=img_ext, 
                                                                                              dataset_type='RTSD')
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.n_classes = self.df['class_name'].nunique() if 'class_name' in self.df else None
        self.n_samples = len(self.df)
        self.mapper = mapper

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):       
        img_name = self.df['file_path'].iloc[index]
        img_path = os.path.join(self.data_dir, img_name)
        image = cv2.imread(img_path)  # BGR
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']
        if self.mapper is not None:
            label = self.df['class_name'].iloc[index]
            return image, self.mapper[label]
        else:
            return image, img_name