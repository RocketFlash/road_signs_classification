import os

# General parameters
PROJECT_NAME = "Road signs classifier"
CUDA_VISIBLE_DEVICES = '0,1,3'
NUM_WORKERS = 10
RANDOM_STATE = 1
USE_WANDB = False
DEBUG = False

# Data 
DATASET_TYPE = 'RTSD'
CLASS_COLUMN = 'class_name'
ID_COLUMN = 'file_path'
DATASET_PATH = '/dataset/RTSD/'
SPLIT_NAME = 'split_simple_1'
TRAIN_IMAGES_PATH = DATASET_PATH
VALID_IMAGES_PATH = DATASET_PATH
TEST_IMAGES_PATH = DATASET_PATH
ANNOS_PATH = os.path.join(DATASET_PATH, 'splits', SPLIT_NAME)
TRAIN_CSV_PATH = os.path.join(ANNOS_PATH, 'train.csv')
VALID_CSV_PATH = os.path.join(ANNOS_PATH, 'val.csv')
WORKDIR_PATH = os.path.join('work_dirs', SPLIT_NAME+'_dist')
WEIGHTS_SAVE_PATH = os.path.join(WORKDIR_PATH, 'weights')
ENCODINGS_SAVE_PATH = os.path.join(WORKDIR_PATH, 'encodings')

# Model
model_name = 'efficientnet-b3'
margin_type = 'arcface'
embeddings_size = 512
scale_size = 64
auto_scale_size = True
loss_type = 'weighted_cross_entropy'
focal_gamma = 2
loss_weights = True
loss_weights_logscale = False
margin_m = 0.1
image_size = 48

# Training
num_epochs = 1000
learning_rate = 0.00001
momentum=0.9
weight_decay=1e-5
train_batch_size = 1000
valid_batch_size = 100
test_batch_size = 100
validate = True
load_margin = True
load_optimizer = False
load_idx_to_class = True
load_best_loss = True
calculate_GAP = False
# RESUME_FROM = None
RESUME_FROM = os.path.join(WEIGHTS_SAVE_PATH, 'best.pt')
WEIGHTS_LOAD_PATH = os.path.join(WEIGHTS_SAVE_PATH, 'best.pt')
# ENCODINGS_LOAD_PATH = None
ENCODINGS_LOAD_PATH = os.path.join(ENCODINGS_SAVE_PATH, 'encodings.pickle')

# Transforms
aug_name = 'medium_rtsd'

# Validation and test
n_top = 10
n_chunks = 20
