# import config_dist_2 as cfg
from configs import config_dist_rtsd as cfg
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICES

import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
import torch.optim as optim 
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from datasets.dataset import get_dataloader
from modules.models import get_model
from modules.metrics import GAP
from modules.transformations import get_transformations
from utils.utils import get_class_mapper, load_ckp, get_loss_weights
from utils.utils_dist import init_process, cleanup
import cv2
from sklearn.metrics.pairwise import cosine_similarity


torch.manual_seed(cfg.RANDOM_STATE)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(cfg.RANDOM_STATE)


def predict_image(model, image_path, device, transform):
    image = cv2.imread(image_path)
    augmented = transform(image=image)
    image = augmented['image']
    image = image.unsqueeze_(0)

    model.eval()
    with torch.no_grad(): 
        image = image.to(device)
        logits = model(image)
        logits = logits.cpu().numpy()
        return logits

def main():
    os.makedirs(cfg.ENCODINGS_SAVE_PATH, exist_ok=True)
    generated_embeddings = {}
    if cfg.ENCODINGS_LOAD_PATH is None:
        print('There is no encodings file, please set it in config file')
    else:
        with open(os.path.join(cfg.ENCODINGS_LOAD_PATH), 'rb') as f:
            generated_embeddings = pickle.load(f)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = get_model(cfg.model_name, embeddings_size=cfg.embeddings_size)
    model = model.to(device)

    print(f'Load checkpoint : {cfg.WEIGHTS_LOAD_PATH}')
    model, _, _, _, _, _ = load_ckp(cfg.WEIGHTS_LOAD_PATH, model,remove_module=True)


    df_train = pd.read_csv(cfg.TRAIN_CSV_PATH)
    df_valid = pd.read_csv(cfg.VALID_CSV_PATH)

    file_idx = 25
    image_path = cfg.DATASET_PATH + df_valid[cfg.ID_COLUMN].iloc[file_idx]
    gt_val = df_valid[cfg.CLASS_COLUMN].iloc[file_idx]

    image_transform = get_transformations(aug_name='transforms_without_aug', image_size=cfg.image_size)
    print(image_path)

    df_train['encodings'] = df_train[cfg.ID_COLUMN].map(generated_embeddings)
    encodings_train = np.array(df_train['encodings'].tolist())
    print(f'Train encodings size {encodings_train.shape}')

    # Get embedding from input image
    encodings_curr = predict_image(model, image_path, device, image_transform['test'])

    cosine_sim_matrix_i = cosine_similarity(encodings_train, encodings_curr)
    total_matrix_idxs = np.arange(cosine_sim_matrix_i.shape[0]).reshape((cosine_sim_matrix_i.shape[0],1)).astype(int)

    res = np.argpartition(cosine_sim_matrix_i, -cfg.n_top, axis=0)[-cfg.n_top:]
    res_vals = np.take_along_axis(cosine_sim_matrix_i, res, axis=0)
    res_idxs = np.take_along_axis(total_matrix_idxs, res, axis=0)
    
    predictions_n = []
    confidences_n = res_vals
    for row_idx, row in enumerate(res_idxs):
        df_prediction = df_train.iloc[row, :]
        pred = np.array(df_prediction[cfg.CLASS_COLUMN].tolist())
        predictions_n.append(pred)
    predictions_n = np.array(predictions_n)

    predictions_final = []
    confidences_final = np.zeros(predictions_n.shape[1])
    for j in range(predictions_n.shape[1]):
        print(predictions_n)
        curr_n_pred = predictions_n[:, j]
        curr_n_conf = confidences_n[:, j]
        dict_conf_list = {}
        dict_conf_sum = {}
        for p, c in zip(curr_n_pred, curr_n_conf):
            if p in dict_conf_list:
                dict_conf_list[p].append(c)
            else:
                dict_conf_list[p] = [c]
        
        dict_conf_sum = { k : sum(v) for k, v in dict_conf_list.items()}
        pred_i = max(dict_conf_sum, key=dict_conf_sum.get)
        conf_i = max(dict_conf_list[pred_i])
        predictions_final.append(pred_i)
        confidences_final[j] = conf_i
    predictions_final = np.array(predictions_final)
    print(f'Prediction class: {predictions_final[0]}')
    print(f'GT class: {gt_val}')
    print(f'Prediction confidence: {confidences_final[0]}')

if __name__=="__main__":
    main()