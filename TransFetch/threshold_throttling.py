from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from numpy import sqrt
from numpy import argmax
import config as cf
from numpy import nanargmax
import numpy as np

BITMAP_SIZE=cf.BITMAP_SIZE
'''
def get_optimal_threshold(y_score,y_real,optimal_type="micro"):
    print("throttleing by optimal threshold")
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    threshold=dict()
    best_threshold=[]
    gmeans=dict()
    ix=dict()
    #pdb.set_trace()
    for i in range(BITMAP_SIZE):
        fpr[i], tpr[i], threshold[i] =roc_curve(y_real[:,i],y_score[:,i])
        roc_auc[i] = auc(fpr[i],tpr[i])
        #best:
        gmeans[i] = sqrt(tpr[i]*(1-fpr[i]))
        ix[i]=argmax(gmeans[i])
        best_threshold.append(threshold[i][ix[i]])
        #print('Dimension: i=%d, Best threshold=%f, G-Mean=%.3f' %(i, threshold[i][ix[i]], gmeans[i][ix[i]]))
    if optimal_type=="indiv":
        return best_threshold
    elif optimal_type=="micro":
        fpr["micro"], tpr["micro"], threshold["micro"] = roc_curve(y_real.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        #best:
        gmeans["micro"] = sqrt(tpr["micro"]*(1-fpr["micro"]))
        ix["micro"]=argmax(gmeans["micro"])
        print('Best micro threshold=%f, G-Mean=%.3f' %(threshold["micro"][ix["micro"]], gmeans["micro"][ix["micro"]]))
        return threshold["micro"][ix["micro"]]
    else:
        return 0.5
'''
def threshold_throttleing(test_df,throttle_type="f1",optimal_type="micro",topk=2,threshold=0.5):
    y_score=np.stack(test_df["y_score"])
    y_real=np.stack(test_df["future"])
    best_threshold=0
    if throttle_type=="roc":
        print("throttleing by roc curve")
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        threshold=dict()
        best_threshold_list=[]
        gmeans=dict()
        ix=dict()
        #pdb.set_trace()
        for i in range(BITMAP_SIZE):
            fpr[i], tpr[i], threshold[i] =roc_curve(y_real[:,i],y_score[:,i])
            roc_auc[i] = auc(fpr[i],tpr[i])
            #best:
            gmeans[i] = sqrt(tpr[i]*(1-fpr[i]))
            ix[i]=nanargmax(gmeans[i])
            best_threshold_list.append(threshold[i][ix[i]])
            #print('Dimension: i=%d, Best threshold=%f, G-Mean=%.3f' %(i, threshold[i][ix[i]], gmeans[i][ix[i]]))
        if optimal_type=="indiv":
            best_threshold=best_threshold_list
            y_pred_bin = (y_score-np.array(best_threshold) >0)*1
            test_df["predicted"]= list(y_pred_bin)#(all,[length])
        elif optimal_type=="micro":
            fpr["micro"], tpr["micro"], threshold["micro"] = roc_curve(y_real.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            #best:
            gmeans["micro"] = sqrt(tpr["micro"]*(1-fpr["micro"]))
            ix["micro"]=nanargmax(gmeans["micro"])
            best_threshold=threshold["micro"][ix["micro"]]
            print('Best micro threshold=%f, G-Mean=%.3f' %(best_threshold, gmeans["micro"][ix["micro"]]))
            
            y_pred_bin = (y_score-best_threshold >0)*1
            test_df["predicted"]= list(y_pred_bin)#(all,[length])
            
    if throttle_type=="f1":
        print("throttleing by precision-recall curve")
        p = dict()
        r = dict()
        threshold=dict()
        best_threshold_list=[]
        fscore=dict()
        ix=dict()
        
        p["micro"], r["micro"], threshold["micro"]=precision_recall_curve(y_real.ravel(),y_score.ravel())
        fscore["micro"] = (2 * p["micro"] * r["micro"]) / (p["micro"] + r["micro"])
        ix["micro"]=nanargmax(fscore["micro"])
        best_threshold=threshold["micro"][ix["micro"]]
        print('Best micro threshold=%f, fscore=%.3f' %(best_threshold, fscore["micro"][ix["micro"]]))
        y_pred_bin = (y_score-best_threshold >0)*1
        test_df["predicted"]= list(y_pred_bin)
        
    elif throttle_type=="topk":
        print("throttleing by topk:",topk)
        pred_index = torch.tensor(y_score).topk(topk)[1].cpu().detach().numpy()
        y_pred_bin=[to_bitmap(a,BITMAP_SIZE) for a in pred_index]
        test_df["predicted"]= list(y_pred_bin)
        
    elif throttle_type =="fixed_threshold":
        print("throttleing by fixed threshold:",threshold)
        best_threshold=threshold
        y_pred_bin = (y_score-np.array(best_threshold) >0)*1
        test_df["predicted"]= list(y_pred_bin)#(all,[length])
    
    return test_df, best_threshold

#%%




