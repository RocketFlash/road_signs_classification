import pandas as pd
import torch
import numpy as np
from collections import OrderedDict

def get_loss_weights(df_path, class_to_idx, logscale=True, class_column='landmark_id',):
    df = pd.read_csv(df_path)
    n_classes = len(class_to_idx)
    n_samples = len(df)
    print(n_classes)
    weights = np.ones(n_classes)
    dict_counts = df[class_column].value_counts().to_dict()
    max_val = max(dict_counts.values())
    for k, v in dict_counts.items():
        if logscale:
            weights[class_to_idx[k]] = 1/np.log(v)
        else:
            weights[class_to_idx[k]] = max_val/v
    return torch.from_numpy(weights).float()

def get_class_mapper(df_path,class_column='landmark_id'):
    class_to_idx = {}
    df = pd.read_csv(df_path)
    classes = list(df[class_column].unique())
    classes = sorted(classes)
    class_to_idx = { c:i for i, c in enumerate(classes)}
    idx_to_class = { i:c for i, c in enumerate(classes)}
    return class_to_idx, idx_to_class

def save_ckp(save_path, epoch, model, margin, optimizer, idx_to_class, best_loss):
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'margin': margin.state_dict(),
        'idx_to_class' : idx_to_class,
        'optimizer': optimizer.state_dict(),
        'best_loss': best_loss
    }
    torch.save(checkpoint, save_path)

def load_ckp(checkpoint_fpath, model, optimizer=None, margin=None, remove_module=False):
    checkpoint = torch.load(checkpoint_fpath)

    pretrained_dict = checkpoint['model']
    model_state_dict = model.state_dict()
    if remove_module:
        new_state_dict = OrderedDict()
        for k, v in pretrained_dict.items():
            name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
            new_state_dict[name] = v
        pretrained_dict = new_state_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_state_dict}
    model_state_dict.update(pretrained_dict)
 

    model.load_state_dict(pretrained_dict)

    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
    except:
        print('Cannot load optimizer params')
        
    if margin is not None and 'margin' in checkpoint:
            margin.load_state_dict(checkpoint['margin'])

    epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
    idx_to_class = checkpoint['idx_to_class'] if 'idx_to_class' in checkpoint else {}
    best_loss = checkpoint['best_loss'] if 'best_loss' in checkpoint else 100
    
    return model, optimizer, margin, epoch, idx_to_class, best_loss