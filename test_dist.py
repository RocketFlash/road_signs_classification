# import config_dist_2 as cfg
# from configs import config_dist_glr as cfg
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
from modules.transformations import get_transformations
from utils.utils import get_class_mapper, load_ckp, get_loss_weights
from utils.utils_dist import init_process, cleanup

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score

torch.manual_seed(cfg.RANDOM_STATE)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(cfg.RANDOM_STATE)


def get_embeddings(rank, world_size):
    generated_embeddings = {}
    init_process(rank, world_size)
    print(f"{rank + 1}/{world_size} process initialized")

    image_transforms = get_transformations(aug_name=cfg.aug_name, image_size=cfg.image_size)
    test_loader, test_data = get_dataloader(data_dir=cfg.TEST_IMAGES_PATH, 
                                            dataset_type=cfg.DATASET_TYPE, 
                                            transform=image_transforms['test'], 
                                            batch_size=cfg.test_batch_size,
                                            num_workers=cfg.NUM_WORKERS,
                                            rank=rank,
                                            world_size=world_size,
                                            drop_last=False)
    if rank==0:
        print(f'Total N testing samples: {test_data.n_samples}')

    model = get_model(cfg.model_name, embeddings_size=cfg.embeddings_size)
    model = model.cuda(rank)
    model = DDP(model, device_ids=[rank])

    print(f'Load checkpoint : {cfg.WEIGHTS_LOAD_PATH}')
    model, _, _, _, _, _ = load_ckp(cfg.WEIGHTS_LOAD_PATH, model)

    model.eval()

    if rank==0:
        tqdm_test = tqdm(test_loader, total=int(len(test_loader)))
    else:
        tqdm_test = test_loader

    with torch.no_grad():
        for batch_index, (data, image_ids) in enumerate(tqdm_test):
            
            data = data.cuda(rank)
            logits = model.module(data)

            logits = logits.cpu().numpy()
            for logits_i, image_id in zip(logits, image_ids):
                generated_embeddings[image_id] = logits_i
    with open(os.path.join(cfg.ENCODINGS_SAVE_PATH, f'encodings_rank_{rank}.pickle'), 'wb') as f:
        pickle.dump(generated_embeddings, f)


def calculate_top_n(sim_matrix,best_top_n_vals,
                               best_top_n_idxs,
                               curr_zero_idx=0,
                               n=10):
    n_rows, n_cols = sim_matrix.shape
    total_matrix_vals = sim_matrix
    total_matrix_idxs = np.tile(np.arange(n_rows).reshape(n_rows,1), (1,n_cols)).astype(int) + curr_zero_idx
    if curr_zero_idx>0:
        total_matrix_vals = np.vstack((total_matrix_vals, best_top_n_vals))
        total_matrix_idxs = np.vstack((total_matrix_idxs, best_top_n_idxs))
    res = np.argpartition(total_matrix_vals, -n, axis=0)[-n:]
    res_vals = np.take_along_axis(total_matrix_vals, res, axis=0)
    res_idxs = np.take_along_axis(total_matrix_idxs, res, axis=0)

    del res, total_matrix_idxs, total_matrix_vals
    return res_vals, res_idxs

def cosine_similarity_chunks(X, Y, n_chunks=5, top_n=5):
    ch_sz = X.shape[0]//n_chunks

    best_top_n_vals = None
    best_top_n_idxs = None

    for i in tqdm(range(n_chunks)):
        chunk = X[i*ch_sz:,:] if i==n_chunks-1 else X[i*ch_sz:(i+1)*ch_sz,:]
        cosine_sim_matrix_i = cosine_similarity(chunk, Y)
        best_top_n_vals, best_top_n_idxs = calculate_top_n(cosine_sim_matrix_i,
                                                           best_top_n_vals,
                                                            best_top_n_idxs,
                                                            curr_zero_idx=(i*ch_sz),
                                                            n=top_n)
    return best_top_n_vals, best_top_n_idxs

def main():
    os.makedirs(cfg.ENCODINGS_SAVE_PATH, exist_ok=True)
    generated_embeddings = {}
    if cfg.ENCODINGS_LOAD_PATH is None:
        N_GPUS = torch.cuda.device_count()
        print(f'N GPUs : {N_GPUS}')
        mp.spawn(get_embeddings,
            args=(N_GPUS,),
            nprocs=N_GPUS,
            join=True)
        
        generated_embeddings_total = {}
        for rank in range(N_GPUS):
            with open(os.path.join(cfg.ENCODINGS_SAVE_PATH, f'encodings_rank_{rank}.pickle'), 'rb') as f:
                generated_embeddings_i = pickle.load(f)
                generated_embeddings_total = {**generated_embeddings_total, **generated_embeddings_i}
        
        print(f'Number of samples to write {len(generated_embeddings_total)}')
        with open(os.path.join(cfg.ENCODINGS_SAVE_PATH, 'encodings.pickle'), 'wb') as f:
            pickle.dump(generated_embeddings_total, f)
        generated_embeddings = generated_embeddings_total
    else:
        with open(os.path.join(cfg.ENCODINGS_LOAD_PATH), 'rb') as f:
            generated_embeddings = pickle.load(f)
    
    df_train = pd.read_csv(cfg.TRAIN_CSV_PATH)
    df_valid = pd.read_csv(cfg.VALID_CSV_PATH)
    df_train['encodings'] = df_train[cfg.ID_COLUMN].map(generated_embeddings)
    df_valid['encodings'] = df_valid[cfg.ID_COLUMN].map(generated_embeddings)
    encodings_train = np.array(df_train['encodings'].tolist())
    encodings_valid = np.array(df_valid['encodings'].tolist())

    print(f'Train encodings size {encodings_train.shape}')
    print(f'Valid encodings size {encodings_valid.shape}')

    best_top_n_vals, best_top_n_idxs = cosine_similarity_chunks(encodings_train, encodings_valid, n_chunks=cfg.n_chunks, top_n=cfg.n_top)
    predictions_n = []
    for row_idx, row in enumerate(best_top_n_idxs):
        df_prediction = df_train.iloc[row, :]
        pred = np.array(df_prediction[cfg.CLASS_COLUMN].tolist())
        predictions_n.append(pred)
    predictions_n = np.array(predictions_n)
    confidences_n = best_top_n_vals

    confidences_final = np.zeros(predictions_n.shape[1])
    predictions_final = []

    for j in range(predictions_n.shape[1]):
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
    gts = np.array(df_valid[cfg.CLASS_COLUMN].tolist())

    print(f'Predictions shape {predictions_final.shape}')
    print(f'GTs shape {gts.shape}')
    print(f'Confidences shape {confidences_final.shape}')
    accuracy = accuracy_score(predictions_final, gts)
    print(f'Accuracy value: {accuracy}')

if __name__=="__main__":
    main()