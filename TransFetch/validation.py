#%% y_train;y_score
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
from preprocessing import read_load_trace_data, preprocessing, to_bitmap
#from torch.autograd import Variable
from sklearn.metrics import roc_curve,f1_score,recall_score,precision_score,accuracy_score
import lzma
from tqdm import tqdm
from data_loader import data_generator,MAPDataset
import os
import pdb
from torch.utils.data import Dataset,DataLoader
import config as cf
from sklearn.metrics import roc_curve, auc
from numpy import sqrt
from numpy import argmax
from threshold_throttling import threshold_throttleing

device=cf.device
batch_size=cf.batch_size
epochs = cf.epochs
lr = cf.lr
gamma = cf.gamma
pred_num=cf.PRED_FORWARD
BLOCK_BITS=cf.BLOCK_BITS
TOTAL_BITS=cf.TOTAL_BITS
LOOK_BACK=cf.LOOK_BACK
PRED_FORWARD=cf.PRED_FORWARD

BLOCK_NUM_BITS=cf.BLOCK_NUM_BITS
PAGE_BITS=cf.PAGE_BITS
BITMAP_SIZE=cf.BITMAP_SIZE
DELTA_BOUND=cf.DELTA_BOUND
SPLIT_BITS=cf.SPLIT_BITS
FILTER_SIZE=cf.FILTER_SIZE


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

#%%

def model_prediction(test_loader, test_df, model_save_path):#"top_k";"degree";"optimal"
    print("predicting")
    prediction=[]
    model.load_state_dict(torch.load(model_save_path))
    model.to(device)
    model.eval()
    y_score=np.array([])
    for data,ip,page,_ in tqdm(test_loader):
        output= model(data,ip,page)
        #prediction.extend(output.cpu())
        prediction.extend(output.cpu().detach().numpy())
    test_df["y_score"]= prediction

    return test_df[['id', 'cycle', 'addr', 'ip','block_address','future', 'y_score']]

def evaluate(y_test,y_pred_bin):
    f1_score_res=f1_score(y_test, y_pred_bin, average='micro')
    #recall: tp / (tp + fn)
    recall_score_res=recall_score(y_test, y_pred_bin, average='micro')
    #precision: tp / (tp + fp)
    precision_score_res=precision_score(y_test, y_pred_bin, average='micro',zero_division=0)
    print("p,r,f1:",precision_score_res,recall_score_res,f1_score_res)
    return precision_score_res,recall_score_res,f1_score_res

##########################################################################################################
#%% New post_processing_delta_bitmap

def convert_hex(pred_block_addr):
    res=int(pred_block_addr)<<BLOCK_BITS
    res2=res.to_bytes(((res.bit_length() + 7) // 8),"big").hex().lstrip('0')
    return res2

def add_delta(block_address,pred_index):
    if pred_index<DELTA_BOUND:
        pred_delta=pred_index+1
    else:
        pred_delta=pred_index-BITMAP_SIZE
        
    return block_address+pred_delta


def post_processing_delta_filter(df):
    print("filtering")
    pred_array=np.stack(df["predicted"])
    pred_n_degree=pred_array
    
    df["pred_index"]=pred_n_degree.tolist()
    df=df.explode('pred_index')
    df=df.dropna()[['id', 'cycle', 'block_address', 'pred_index']]
    
    #add delta to block address
    df['pred_block_addr'] = df.apply(lambda x: add_delta(x['block_address'], x['pred_index']), axis=1)
    
    #filter
    que = []
    pref_flag=[]
    dg_counter=0
    df["id_diff"]=df["id"].diff()
    
    for index, row in df.iterrows():
        if row["id_diff"]!=0:
            que.append(row["block_address"])
            dg_counter=0
        pred=row["pred_block_addr"]
        if dg_counter<cf.Degree:
            if pred in que:
                pref_flag.append(0)
            else:
                que.append(pred)
                pref_flag.append(1)
                dg_counter+=1
        else:
            pref_flag.append(0)
        que=que[-FILTER_SIZE:]
    
    df["pref_flag"]=pref_flag
    df=df[df["pref_flag"]==1]
    df['pred_hex'] = df.apply(lambda x: convert_hex(x['pred_block_addr']), axis=1)
    df_res=df[["id","pred_hex"]]
    return df_res
    

def run_val(test_loader,test_df,file_path,model_save_path):
    print("Validation start")
    test_df=model_prediction(test_loader, test_df,model_save_path)

    df_thresh={}
    app_name=file_path.split("/")[-1].split("-")[0]
    val_res_path=model_save_path+".val_res.csv"
    
    df_res, threshold=threshold_throttleing(test_df,throttle_type="f1",optimal_type="micro")
    p,r,f1 = evaluate(np.stack(df_res["future"]), np.stack(df_res["predicted"]))
    df_thresh["app"],df_thresh["opt_th"],df_thresh["p"],df_thresh["r"],df_thresh["f1"]=[app_name],[threshold],[p],[r],[f1]
    
    df_res, _ =threshold_throttleing(test_df,throttle_type="fixed_threshold",threshold=0.5)
    p,r,f1 = evaluate(np.stack(df_res["future"]), np.stack(df_res["predicted"]))
    df_thresh["p_5"],df_thresh["r_5"],df_thresh["f1_5"]=[p],[r],[f1]
    
    pd.DataFrame(df_thresh).to_csv(val_res_path,header=1, index=False, sep=" ") #pd_read=pd.read_csv(val_res_path,header=0,sep=" ")
    print("Done: results saved at:", val_res_path)

#%%
if __name__ == "__main__":
    file_path=sys.argv[1]
    model_save_path=sys.argv[2]
    TRAIN_NUM = int(sys.argv[3])
    TOTAL_NUM = int(sys.argv[4])
    SKIP_NUM = int(sys.argv[5])
    test_loader, test_df = data_generator(file_path,TRAIN_NUM,TOTAL_NUM,SKIP_NUM,only_val=True)
    run_val(test_loader,test_df,file_path,model_save_path)