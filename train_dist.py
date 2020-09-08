from configs import config_dist_rtsd as cfg
# from configs import config_dist_glr as cfg
# import config_dist_2 as cfg
import os
from shutil import copyfile

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICES

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from torch.optim.lr_scheduler import CyclicLR, StepLR, ReduceLROnPlateau
from datasets.dataset import get_dataloader
from modules.models import get_model
from modules.margins import get_margin
from modules.metrics import accuracy
from modules.losses import get_loss
from modules.transformations import get_transformations

from utils.utils import get_class_mapper, load_ckp, get_loss_weights, save_ckp
from utils.utils_dist import init_process, cleanup

torch.manual_seed(cfg.RANDOM_STATE)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(cfg.RANDOM_STATE)

WANDB_AVAILABLE = False
if cfg.DEBUG:
    best_weights_name = 'debug_best.pt'
    last_weights_name = 'debug_last.pt'
else:
    best_weights_name = 'best.pt'
    last_weights_name = 'last.pt'
    
if cfg.USE_WANDB:
    try:
        import wandb
        WANDB_AVAILABLE = True
    except ModuleNotFoundError:
        print('wandb isnot installed')

def train_epoch(model, margin, train_loader, optimizer, scaler, loss_func, rank, epoch=0):
    train_loss = 0.0
    train_acc = 0.0

    model.train()

    if rank==0:
        tqdm_train = tqdm(train_loader, total=int(len(train_loader)))
    else:
        tqdm_train = train_loader
    idx1 = 0
    for batch_index, (data, target) in enumerate(tqdm_train):
        if cfg.DEBUG:
            if idx1>=10:
                break
            idx1+=1
        data = data.cuda(rank)
        target = target.cuda(rank)

        optimizer.zero_grad()

        with amp.autocast(enabled=True):
            logits = model(data)
            output = margin(logits, target)
            loss = loss_func(output, target)
            acc = accuracy(output, target) * 100
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * data.size(0)
        train_acc += acc * data.size(0)
        if rank==0:
            tqdm_train.set_postfix(epoch=epoch, train_acc=train_acc/((batch_index+1) * train_loader.batch_size), 
                                                train_loss=(train_loss / ((batch_index+1) * train_loader.batch_size)))

    return train_loss, train_acc

def valid_epoch(model, margin, valid_loader, loss_func, epoch=0):
    valid_loss = 0.0
    valid_acc = 0.0
    model.eval()

    activation = nn.Softmax(dim=1)
    device = next(model.parameters()).device
    tqdm_val = tqdm(valid_loader, total=int(len(valid_loader)))
    vals_gt, vals_pred, vals_conf = [], [], []

    idx1 = 0
    with torch.no_grad():
        for batch_index, (data, target) in enumerate(tqdm_val):
            if cfg.DEBUG:
                if idx1>60:
                    break
            idx1+=1
            data = data.to(device)
            target = target.to(device)
            
            logits = model.module(data)
            output = margin.module(logits, target)

            if cfg.calculate_GAP:
                output_probs = activation(output)
                confs, pred = torch.max(output_probs, dim=1)

                vals_conf.extend(confs.cpu().numpy().tolist())
                vals_pred.extend(pred.cpu().numpy().tolist())
                vals_gt.extend(target.cpu().numpy().tolist())

            loss = loss_func(output, target)
            acc = accuracy(output, target) * 100

            valid_loss += loss.item() * data.size(0)
            valid_acc += acc * data.size(0)

            tqdm_val.set_postfix(epoch=epoch, val_acc=valid_acc/((batch_index+1) * valid_loader.batch_size),
                                        val_loss=(valid_loss / ((batch_index+1) * valid_loader.batch_size)))
    

    return valid_loss, valid_acc


def train(rank, world_size):
    init_process(rank, world_size)
    print(f"{rank + 1}/{world_size} process initialized")

    epoch_i = 0
    image_transforms = get_transformations(aug_name=cfg.aug_name, image_size=cfg.image_size)
    class_to_idx, idx_to_class = get_class_mapper(cfg.TRAIN_CSV_PATH, class_column=cfg.CLASS_COLUMN)
    train_loader, train_data = get_dataloader(data_dir=cfg.TRAIN_IMAGES_PATH,
                                              df_path=cfg.TRAIN_CSV_PATH,  
                                              dataset_type=cfg.DATASET_TYPE,
                                              mapper=class_to_idx, 
                                              transform=image_transforms['train'], 
                                              batch_size=cfg.train_batch_size,
                                              rank=rank,
                                              world_size=world_size,
                                              num_workers=cfg.NUM_WORKERS)
    scale_size = np.sqrt(2) *np.log(train_data.n_classes-1) if cfg.auto_scale_size else cfg.scale_size
    if rank==0:
        if cfg.USE_WANDB and WANDB_AVAILABLE and not cfg.DEBUG:
            wandb.init(project=cfg.PROJECT_NAME)
        valid_loader, valid_data = get_dataloader(data_dir=cfg.VALID_IMAGES_PATH,
                                                  df_path=cfg.VALID_CSV_PATH,
                                                  dataset_type=cfg.DATASET_TYPE,  
                                                  mapper=class_to_idx, 
                                                  transform=image_transforms['valid'], 
                                                  batch_size=cfg.valid_batch_size,
                                                  num_workers=cfg.NUM_WORKERS)
        print(f'Total N classes: {train_data.n_classes}')
        print(f'Total N classes val: {valid_data.n_classes}')
        print(f'Total N training samples: {train_data.n_samples}')
        print(f'Total N validation samples: {valid_data.n_samples}')
        print(f'Scale size s : {scale_size:.2f}')

 

    model = get_model(cfg.model_name, embeddings_size=cfg.embeddings_size)
    margin = get_margin(cfg.margin_type, 
                        cfg.embeddings_size, 
                        train_data.n_classes, scale_size=scale_size, m=cfg.margin_m)

    model = model.cuda(rank)
    margin = margin.cuda(rank)

    model = DDP(model, device_ids=[rank])
    margin = DDP(margin, device_ids=[rank])

    loss_weights =  None
    if cfg.loss_weights:
        loss_weights = get_loss_weights(cfg.TRAIN_CSV_PATH, class_to_idx, cfg.loss_weights_logscale, class_column=cfg.CLASS_COLUMN)

    loss_func = get_loss(cfg.loss_type, weight=loss_weights,gamma=cfg.focal_gamma)
    loss_func = loss_func.cuda(rank)

    optimizer = optim.SGD([{'params': model.parameters(), 'weight_decay': cfg.weight_decay},
                            {'params': margin.parameters(), 'weight_decay': cfg.weight_decay}], 
                            lr=world_size*cfg.learning_rate,
                            momentum=cfg.momentum, 
                            nesterov=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    best_loss = 100

    if cfg.RESUME_FROM is not None:
        if rank==0:
            print(f'Resume training from: {cfg.RESUME_FROM}')
        
        mrgn = margin if cfg.load_margin else None
        optmzr = optimizer if cfg.load_optimizer else None
        model, optmzr, mrgn, epoch_i, idx_2_class, bst_loss = load_ckp(cfg.RESUME_FROM, model, optmzr, mrgn)
        margin = mrgn if cfg.load_margin else margin
        optimizer = optmzr if cfg.load_optimizer else optimizer
        idx_to_class = idx_2_class if cfg.load_idx_to_class else idx_to_class
        best_loss = bst_loss if cfg.load_best_loss else 100

    
    if rank==0:
        print(f'Current best loss: {best_loss}')
        if cfg.USE_WANDB and WANDB_AVAILABLE and not cfg.DEBUG:
            wandb.watch(model)
            wandb.watch(margin)

    
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    metrics = {}
    scaler = amp.GradScaler(enabled=True)

    for epoch in range(1, cfg.num_epochs + 1):
        if epoch<=epoch_i:
            continue
        train_loss, train_acc = train_epoch(model, margin, train_loader, optimizer, scaler, loss_func=loss_func, rank=rank, epoch=epoch)
        train_loss = train_loss/len(train_loader.sampler)
        train_acc = train_acc/len(train_loader.sampler)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        if rank==0:
            cpk_save_path = os.path.join(cfg.WEIGHTS_SAVE_PATH, last_weights_name)
            save_ckp(cpk_save_path, epoch, model, margin, optimizer, idx_to_class, best_loss)
            
            metrics['train_loss'] = train_loss
            metrics['train_acc'] = train_acc
            metrics['learning_rate'] = optimizer.param_groups[0]['lr']
            
            epoch_info_str = f'Epoch: {epoch} Training Loss: {train_loss:.5f} Training Acc  {train_acc:.5f}\n'
            if cfg.validate:
                valid_loss, valid_acc = valid_epoch(model, margin, valid_loader, loss_func=loss_func, epoch=epoch)
                valid_loss = valid_loss/len(valid_loader.sampler)
                valid_acc = valid_acc/len(valid_loader.sampler)
                valid_losses.append(valid_loss)
                valid_accs.append(valid_acc)

                metrics['valid_loss'] = valid_loss
                metrics['valid_acc'] = valid_acc

                scheduler.step(valid_loss)

                if valid_loss < best_loss:
                    print('Saving best model')
                    best_loss = valid_loss
                    cpk_save_path = os.path.join(cfg.WEIGHTS_SAVE_PATH, best_weights_name)
                    save_ckp(cpk_save_path, epoch, model, margin, optimizer, idx_to_class, best_loss)
                epoch_info_str += f'         Validation Loss: {valid_loss:.5f} Validation Acc  {valid_acc:.5f}'
            else:
                scheduler.step(train_loss)

                if train_loss < best_loss:
                    print('Saving best model')
                    best_loss = train_loss
                    cpk_save_path = os.path.join(cfg.WEIGHTS_SAVE_PATH, best_weights_name)
                    save_ckp(cpk_save_path, epoch, model, margin, optimizer, idx_to_class, best_loss)

            if cfg.USE_WANDB and not cfg.DEBUG:
                wandb.log(metrics, step=epoch)

            # print training/validation statistics
            print(epoch_info_str)
        dist.barrier()
    cleanup()

def main():
    if cfg.DEBUG:
        print('DEBUG MODE')
    os.makedirs(cfg.WEIGHTS_SAVE_PATH, exist_ok=True)
    N_GPUS = torch.cuda.device_count()
    print(f'N GPUs : {N_GPUS}')
    mp.spawn(train,
        args=(N_GPUS,),
        nprocs=N_GPUS,
        join=True)

if __name__=="__main__":
    main()