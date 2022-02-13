#%% y_train;y_score
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'#'2 in tarim'
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
from preprocessing import read_load_trace_data, preprocessing, to_bitmap,preprocessing_gen
from torch.autograd import Variable
from sklearn.metrics import roc_curve,f1_score,recall_score,precision_score,accuracy_score
import lzma
from tqdm import tqdm
from data_loader import data_generator,MAPDataset,data_generator_gen,MAPDataset_gen
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
LAY_BACK=50
#cf.LAY_BACK


#%%

def model_prediction_gen(test_loader, test_df, model_save_path):#"top_k";"degree";"optimal"
    print("predicting")
    prediction=[]
    model.load_state_dict(torch.load(model_save_path))
    model.to(device)
    model.eval()
    y_score=np.array([])
    for data,ip,page in tqdm(test_loader):
        output= model(data,ip,page)
        #prediction.extend(output.cpu())
        prediction.extend(output.cpu().detach().numpy())
    test_df["y_score"]= prediction

    return test_df[['id', 'cycle', 'addr', 'ip','block_address', 'y_score']]

def evaluate(y_test,y_pred_bin):
    f1_score_res=f1_score(y_test, y_pred_bin, average='samples')
    #recall: tp / (tp + fn)
    recall_score_res=recall_score(y_test, y_pred_bin, average='samples')
    #precision: tp / (tp + fp)
    precision_score_res=precision_score(y_test, y_pred_bin, average='samples',zero_division=0)
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

def bitmap_to_index_list(y_score,threshold):
    sorted_pred_index=torch.tensor(y_score).topk(len(np.where([y_score>=threshold])[1]))[1].numpy()
    #return sorted index
    return sorted_pred_index

def post_processing_delta_filter(df,opt_threshold,filtering=True):
    print("post_processing")
    df["pred_index"]=df.apply(lambda x: bitmap_to_index_list(x['y_score'], opt_threshold), axis=1)
    df=df.explode('pred_index')
    df=df.dropna()[['id', 'cycle', 'block_address', 'pred_index']]
    #add delta to block address
    df['pred_block_addr'] = df.apply(lambda x: add_delta(x['block_address'], x['pred_index']), axis=1)
    
    if filtering==True:
        #filter
        print("filtering")
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
    #df=df[["id","pred_hex"]]
    return df

#%%
def degree_stats(df,app_name,degree_stats_path):
    dic_dgsts={}
    dfc=df.groupby(["id"]).size().reset_index(name='counts')
    dfc=dfc.agg(["mean","max","min","median"])
    dic_dgsts["app"],dic_dgsts["mean"],dic_dgsts["max"],dic_dgsts["min"],dic_dgsts["median"]=\
        [app_name],[dfc['counts']["mean"]],[dfc['counts']["max"]],[dfc['counts']["min"]],[dfc['counts']["median"]]
    df_sts=pd.DataFrame(dic_dgsts)
    pd.DataFrame(df_sts).to_csv(degree_stats_path,header=1, index=False, sep=" ") #pd_read=pd.read_csv(val_res_path,header=0,sep=" ")
    print(df_sts)
    print("Done: results saved at:", degree_stats_path)
    

def induce_latency(df,latency=LAY_BACK):
    print("")
    print("inducing latency:",latency)
    pref_with_latency=[1]
    hist_cycle=0
    begin=True
    for index, row in df.iterrows():
        if begin==True:
            hist_cycle=row["cycle"]
            begin=False
        else:
            current_cycle=row["cycle"]
            if current_cycle>hist_cycle+latency:
                pref_with_latency.append(1)
                hist_cycle=current_cycle
            else:
                pref_with_latency.append(0)

    df["pref_with_latency"]=pref_with_latency
    return df[df["pref_with_latency"]>0]


#%%
'''
file_path="/home/pengmiao/Disk/work/HPCA/ML-DPC-S0/LoadTraces/spec17/654.roms-s0.txt.xz"
model_save_path="/home/pengmiao/Disk/work/MLSYS2022/results/654.model.pth"
#file_path="/home/pengmiao/Disk/work/MLPrefComp/ChampSim/ML-DPC/LoadTraces/spec06/429.mcf-s0.txt.xz"
#model_save_path="./429.mcf-s0.txt.xz.model.pth"

SKIP_NUM=1
TRAIN_NUM = 3
TOTAL_NUM=5

'''

if __name__ == "__main__":
    file_path=sys.argv[1]
    model_save_path=sys.argv[2]
    TRAIN_NUM = int(sys.argv[3])
    TOTAL_NUM = int(sys.argv[4])
    SKIP_NUM = int(sys.argv[5])
    val_res_path=model_save_path+".val_res.csv"
    
    path_to_prefetch_file = model_save_path+".prefetch_file_dist_50.csv"
    degree_stats_path=model_save_path+".degree_stats_dist_50.csv"
    print("Generation with latency start")
    opt_threshold=pd.read_csv(val_res_path,header=0,sep=" ")["opt_th"].values[0]
    
    test_loader, test_df = data_generator_gen(file_path,TRAIN_NUM,TOTAL_NUM,SKIP_NUM)
    test_df=model_prediction_gen(test_loader, test_df,model_save_path)
    test_df=induce_latency(test_df)
    test_df=post_processing_delta_filter(test_df,opt_threshold,filtering=True)
    test_df[["id","pred_hex"]].to_csv(path_to_prefetch_file,header=False, index=False, sep=" ")
    print("prefetch file saved")
    app_name=file_path.split("/")[-1].split("-")[0]
    degree_stats_path=model_save_path+".degree_stats_dist_50.csv"
    
    degree_stats(test_df[["id"]],app_name,degree_stats_path)

