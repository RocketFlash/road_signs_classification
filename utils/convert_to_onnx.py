import os
import sys
sys.path.append('..')
from configs import config_dist_rtsd as cfg
from efficientnet_pytorch import EfficientNet

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICES

from modules.models import get_model
from utils import load_ckp
import torch.onnx
import onnx
import torch

OPSET_VERSION = 8
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = get_model(cfg.model_name, embeddings_size=cfg.embeddings_size)
model = model.to(device)
model.effnet.set_swish(False)

print(f'Load checkpoint : {cfg.WEIGHTS_LOAD_PATH}')
model, _, _, _, _, _ = load_ckp('../' + cfg.WEIGHTS_LOAD_PATH, model,remove_module=True)
model.eval()

with torch.no_grad(): 
    # Input to the model
    x = torch.randn(10, 3, 48, 48).to(device).float()
    print('Start onnx conversion')
    print(f'Save model to : {cfg.WEIGHTS_SAVE_PATH}')
    torch.onnx.export(model, x, os.path.join('..', cfg.WEIGHTS_SAVE_PATH, 'model.onnx'), 
                      opset_version=OPSET_VERSION, 
                      verbose=False,
                      export_params=True,
                      input_names = ['input'],
                      output_names = ['output'],
                      dynamic_axes={'input' : {0 : 'batch_size'},    
                                    'output' : {0 : 'batch_size'}})
print('Conversion completed!')

onnx_model = onnx.load(os.path.join('..', cfg.WEIGHTS_SAVE_PATH, 'model.onnx'))
onnx.checker.check_model(onnx_model)

onnx.helper.printable_graph(onnx_model.graph)