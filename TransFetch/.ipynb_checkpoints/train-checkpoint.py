import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'#'2 in tarim'
import warnings
warnings.filterwarnings('ignore')
import sys
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import pandas as pd
import config as cf
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from model import TMAP,TMAP_C
#from preprocessing import read_load_trace_data,preprocessing_patch
from torch.autograd import Variable
from sklearn.metrics import roc_curve,f1_score,recall_score,precision_score,accuracy_score
import lzma
from tqdm import tqdm
from data_loader import data_generator
import os
import pdb
from validation import run_val


device=cf.device
batch_size=cf.batch_size
epochs = cf.epochs
lr = cf.lr
gamma = cf.gamma
step_size=cf.step_size
pred_num=cf.PRED_FORWARD
early_stop = cf.early_stop

#%%

model = TMAP_C(
    image_size=cf.image_size,
    patch_size=cf.patch_size,
    num_classes=cf.num_classes,
    dim=cf.dim,
    depth=cf.depth,
    heads=cf.heads,
    mlp_dim=cf.mlp_dim,
    channels=cf.channels,
    context_gamma=cf.context_gamma
).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
log=cf.Logger()
#%%

def train(ep,train_loader,model_save_path):
    global steps
    epoch_loss = 0
    model.train()
    for batch_idx, (data, ip, page, target)in enumerate(train_loader):#d,t: (torch.Size([64, 1, 784]),64)        
        optimizer.zero_grad()
        output = model(data,ip, page)
        #loss = F.binary_cross_entropy_with_logits(output, target)
        loss = F.binary_cross_entropy(output, target,reduction='mean')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss/=len(train_loader)
    return epoch_loss


def test(test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, ip, page, target in test_loader:
            output = model(data,ip, page)
            test_loss += F.binary_cross_entropy(output, target, reduction='mean').item()
            thresh=0.5
            #thresh=output.data.topk(pred_num)[0].min(1)[0].unsqueeze(1)
            output_bin=(output>=thresh)*1
            correct+=(output_bin&target.int()).sum()
        
        test_loss /=  len(test_loader)
        return test_loss   

def run_epoch(epochs, loading, model_save_path,train_loader,test_loader,lr):
    if loading==True:
        model.load_state_dict(torch.load(model_save_path))
        log.logger.info("-------------Model Loaded------------")
        
    best_loss=0
    early_stop=cf.early_stop
    model.to(device)
    for epoch in range(epochs):
        train_loss=train(epoch,train_loader,model_save_path)
        test_loss=test(test_loader)
        log.logger.info((f"Epoch: {epoch+1} - loss: {train_loss:.10f} - test_loss: {test_loss:.10f}"))
        if epoch == 0:
            best_loss=test_loss
        if test_loss<=best_loss:
            torch.save(model.state_dict(), model_save_path)    
            best_loss=test_loss
            log.logger.info("-------- Save Best Model! --------")
            early_stop=cf.early_stop
        else:
            early_stop-=1
            log.logger.info("Early Stop Left: {}".format(early_stop))
        if early_stop == 0:
            log.logger.info("-------- Early Stop! --------")
            break
            
        #test(test_loader)
        #scheduler.step()
        
        
        
#%%
##########################################################################################################

if __name__ == "__main__":

    file_path=sys.argv[1]
    model_save_path=sys.argv[2]
    TRAIN_NUM = int(sys.argv[3])
    TOTAL_NUM = int(sys.argv[4])
    SKIP_NUM = int(sys.argv[5])

    if os.path.isfile(model_save_path) :
       loading=True
    else:
       loading=False
       
    log_path=model_save_path+".log"
    log.set_logger(log_path)
    log.logger.info("%s"%file_path)
    train_loader, test_loader, test_df = data_generator(file_path,TRAIN_NUM,TOTAL_NUM,SKIP_NUM)
    log.logger.info("-------------Data Proccessed------------")
    run_epoch(epochs, loading,model_save_path,train_loader,test_loader,lr=cf.lr)
    run_val(test_loader,test_df,file_path,model_save_path)
