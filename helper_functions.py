# useful functions

import numpy as np
from sklearn.model_selection import StratifiedKFold
import numpy.matlib
import sklearn
import random


def splitDataManuel(train_ratio):
    # train_ratio should be between 0 and 1
    N0_train = math.floor(N0*train_ratio);
    N0_test = N0-N0_train;
    N1_train = math.floor(N1*train_ratio);
    N1_test = N1-N1_train;
    N_train = N0_train+N1_train
    N_test = N-N_train
    i0 = np.nonzero(label==0)[0];
    i1 = np.nonzero(label==1)[0];
    i0_train = np.random.choice(i0, N0_train, replace=False);
    i1_train = np.random.choice(i1, N1_train, replace=False);
    i0_test = np.setdiff1d(i0, i0_train);
    i1_test = np.setdiff1d(i1, i1_train);
    
    i_train = np.r_[i0_train, i1_train];
    y_train = np.r_[label[i0_train], label[i1_train]];
    i_test = np.r_[i0_test, i1_test]
    y_test = np.r_[label[i0_test], label[i1_test]];
    return (i_train, y_train, i_test, y_test)


def splitFix(folds, label):
    cv = StratifiedKFold(n_splits=folds)
    for (i_train, i_test) in cv.split(np.zeros(N), label):
        break
    return (i_train, i_test)


# padding sequences of arbitrary dimensions
def toFixedLength(stream, limit, pad_dir='left'):
    N = stream.shape[0]
    dim = stream[0].shape[1:]
    output = np.zeros((N,limit)+dim)
    for i in range(N):
        temp = stream[i]
        size_t = temp.shape[0]
        if size_t<limit:
            if pad_dir=='left':
                temp = np.concatenate((np.zeros((limit-size_t,)+dim),temp), axis=0)
            else:
                temp = np.concatenate((temp, np.zeros((limit-size_t,)+dim)), axis=0)
        else:
            temp = temp[-limit:,...]
        output[i,...] = temp
    return output


def splitData(i_train, i_test, max_length_train, pad_dir='left'):
    y_train = label[i_train]
    y_test = label[i_test]
    x_con_train = timeSeriesCon[i_train]
    x_con_test = timeSeriesCon[i_test]
    x_train = toFixedLength(x_con_train, max_length_train, pad_dir)
    max_length_test = np.max(sizes2[i_test])
    x_test = toFixedLength(x_con_test, max_length_test, pad_dir)
#    y_test_repeat = np.matlib.repmat(y_test,max_length_test,1)
#    y_test_repeat = y_test_repeat.reshape((-1, max_length_test,1))
    return (x_train, y_train, x_test, y_test, max_length_test)


def splitDataByCP(i_train, i_test, cps):
    N_train = i_train.size
    y_train = label[i_train]
    y_test = label[i_test]
    x_con_train = timeSeriesCon[i_train]
    x_con_test = timeSeriesCon[i_test]
    sizes = np.zeros(N_train,dtype=int)
    for i in range(N_train):
        index = i_train[i]
        last_cp = np.array(cps[index][2], dtype=int)[-2]-1
        x_con_train[i] = x_con_train[i][last_cp:,:]
        sizes[i] = x_con_train[i].shape[0]
    max_length_train = sizes.max()
    x_train = toFixedLength(x_con_train, max_length_train)
    max_length_test = np.max(sizes2[i_test])
    x_test = toFixedLength(x_con_test, max_length_test)
    return (x_train, y_train, x_test, y_test, max_length_train, max_length_test)


def splitDataByTime(i_train, i_test, time):
    N_train = i_train.size
    y_train = label[i_train]
    y_test = label[i_test]
    x_con_train = timeSeriesCon[i_train]
    x_con_test = timeSeriesCon[i_test]
    if(time>0):
        sizes = np.zeros(N_train,dtype=int)
        for i in range(N_train):
            index = i_train[i]
            start = findIndex(time,index,timeSeries)
            x_con_train[i] = x_con_train[i][start:,:]
            sizes[i] = x_con_train[i].shape[0]
        
    else:
        sizes = sizes2[i_train]
    max_length_train = sizes.max()
    x_train = toFixedLength(x_con_train, max_length_train)
    max_length_test = np.max(sizes2[i_test])
    x_test = toFixedLength(x_con_test, max_length_test)
    return (x_train, y_train, x_test, y_test, max_length_train, max_length_test)



def findIndex(time_ahead, time_axis):
    temp = np.nonzero(time_axis+time_ahead<=0)[0]
    if temp.size>=1:
        return temp[-1]
    else:
        return 0


def getScores(risk, time_ahead, sizes2, i_test, pad_dir='left', dataset=1): # just for regular
    N_test = risk.shape[0]
    scores = np.zeros(N_test)
    max_length_test = np.max(sizes2[i_test])
    for i in range(N_test):
        index = i_test[i]
        start = max_length_test - sizes2[index] if pad_dir=='left' else 0
        if(dataset==1):
            time_axis = np.arange(-sizes2[index]+1,1)*2
        else:
            time_axis = np.arange(-sizes2[index]+1,1)
        end = start + findIndex(time_ahead, time_axis) + 1
        scores[i] = np.max(risk[i, start:end])
    return scores


# without using timeSeries
def getScores2(risk, pad_dir='left', window=0):
    N_test = risk.shape[0]
    scores = np.zeros(N_test)
    max_length_test = np.max(sizes2[i_test])
    for i in range(N_test):
        index = i_test[i]
        start = max_length_test - sizes2[index] if pad_dir=='left' else 0
        end = start + sizes2[index]
        if(window!=0):
            start = np.maximum(end-window, start)
        scores[i] = np.max(risk[i, start:end])
    return scores


def whole_metric(risk, y_test, window=24, window_to_consider=0): # pad_dir='right'
    FAR = []
    TAR = []
    PTA = []
    N_test = risk.shape[0]
    whole_label = np.zeros_like(risk, dtype=int)
    for i in range(N_test):
        if(y_test[i]==1):
            index = i_test[i]
            start = max(sizes2[index]-window, 0)
            whole_label[i, start:sizes2[index]] = 1
    thresholds = np.arange(0,1,0.01)
    for j in range(thresholds.size):
        whole_predict = np.zeros_like(risk, dtype=int)
        for i in range(N_test):
            whole_predict[i, risk[i]>=thresholds[j]] = 1
        TP = 0
        FN = 0
        FP = 0
        TN = 0
        for i in range(N_test):
            _range = sizes2[i_test[i]]
            st = np.maximum(_range-window_to_consider, 0) if(window_to_consider!=0) else 0
            TP += np.sum(np.multiply(whole_label[i,st:_range], whole_predict[i,st:_range]))
            FN += np.sum(np.multiply(whole_label[i,st:_range], 1-whole_predict[i,st:_range]))
            FP += np.sum(np.multiply(1-whole_label[i,st:_range], whole_predict[i,st:_range]))
            TN += np.sum(np.multiply(1-whole_label[i,st:_range], 1-whole_predict[i,st:_range]))
        # ensure no nan happens
        FAR.append(FP/np.maximum(FP+TN, 1))
        TAR.append(TP/np.maximum(TP+FN, 1))
        PTA.append(TP/np.maximum(FP+TP, 1))
    auc_tpr_fpr = sklearn.metrics.auc(FAR,TAR)
    auc_tpr_ppv = sklearn.metrics.auc(TAR,PTA)
    return auc_tpr_fpr, auc_tpr_ppv, window, FAR, TAR, PTA
        
        

def plotAUC(ax1, ax2, indices, aucmean1, aucmean2, x_axis=None, nb_obs=None):
    if nb_obs is None:
        nb_obs = auc1[indices[0]].shape[1]
    if x_axis is None:
        x_axis = np.arange(5,1+5*nb_obs,5)
    if(x_axis.dtype!="object"): # if object, then different x_axis for different axis
        x_axis = np.matlib.repmat(x_axis, len(indices), 1)
    j = 0
    for i, color in zip(indices, colors):
        ax1.plot(x_axis[j], aucmean1[i][:x_axis[j].size], lw=lw, color=color, label=labels[i])
        ax2.plot(x_axis[j], aucmean2[i][:x_axis[j].size], lw=lw, color=color, label=labels[i])
        j += 1
    
    
def addInfo(ax, title):
    ax.legend(loc="lower right")
    ax.set_xlabel("number of epochs")
    ax.set_ylabel("AUC")
    ax.set_title(title)


def plotSelected(indices, name="", x_axis=None, nb_obs=None):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
#    test = np.vectorize(lambda x: np.any(np.isnan(x)))
#    if(test(tcost_mean[indices])):
#        fig3 = plt.figure()
#        ax3 = fig3.add_subplot(111)
#        ax3.plot
    plotAUC(ax1, ax2, indices, auc1_mean, auc2_mean, x_axis, nb_obs)
    addInfo(ax1, "AUC (TPR-FPR), "+name)
    addInfo(ax2, "AUC (TPR-PPV), "+name)
#    indices_34 = []
#    for i in indices:
#        if(auc3[i].size>0):
#            indices_34.append(i)
#    if(len(indices_34)>0):
#        fig3 = plt.figure()
#        ax3 = fig3.add_subplot(111)
#        fig4 = plt.figure()
#        ax4 = fig4.add_subplot(111)
#        plotAUC(ax3, ax4, indices_34, auc3_mean, auc4_mean, x_axis, nb_obs)
#        addInfo(ax3, "whole AUC (TPR-FPR), "+name)
#        addInfo(ax4, "whole AUC (TPR-PPV), "+name)
    return (ax1, ax2)


# create sub sequences using one cps object
def divSeq(stream, cps):
    cp = np.array(cps[2],dtype=int)-1
    nb_c = np.array(cps[0],dtype=int)[0]
    temp = np.empty(nb_c, dtype=object)
    for j in range(nb_c):
        temp[j] = stream[cp[j]:cp[j+1],:]
    return temp

# create sub sequences for a series of time series
def divideSeq(cps, stream):
    N = len(cps)
    output = np.empty(N, dtype=object)
    for i in range(N):
        a = stream[i]
        output[i] = divSeq(a, cps[i])
    if N==1:
        output = output[0]
    return output

def constrained_sum_sample_pos(n, total):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""

    dividers = sorted(random.sample(range(1, total), n - 1))
    return [a - b for a, b in zip(dividers + [total], [0] + dividers)]

