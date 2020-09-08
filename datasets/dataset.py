import pandas as pd
import os
import glob

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def generate_test_df(data_dir, column_name='id', img_ext='.jpg', dataset_type='RTSD'):
    
    all_files = glob.glob(data_dir + f'**/*{img_ext}',recursive=True)
    if dataset_type=='RTSD':
        files = ['/'.join(f.split('/')[-3:]) for f in all_files]
    return pd.DataFrame(files, columns=[column_name])


def get_dataloader(data_dir, dataset_type='RTSD', transform=None, batch_size=8, df_path=None, 
                             mapper=None, rank=-1, world_size=-1, num_workers=8, 
                             pin_memory=True, drop_last=True):
    dataset, sampler = None, None
    if dataset_type=='RTSD':
        from .RTSD import RTSDDataset
        dataset = RTSDDataset(data_dir=data_dir, df_path=df_path, mapper=mapper, transform=transform)
    if rank>=0 and world_size>=0:
        sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=True)
    data_loader = DataLoader(dataset=dataset, 
                             batch_size=batch_size, 
                             sampler=sampler, 
                             pin_memory=pin_memory,
                             drop_last=drop_last,
                             num_workers=num_workers)
    return data_loader, dataset