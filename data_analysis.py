"""
Analysis of MIMIC-III dataset
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pickle_utils as pu
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from scipy.stats import pearsonr
from smooth import savitzky_golay
from scipy.interpolate import interp1d
from scipy.io import savemat
from helper_function import *

timeSeriesCon2, label2, sizes2_2, _ = pu.load('dataset/data.pkl.gz');
N2 = timeSeriesCon2.shape[0]
dim2 = timeSeriesCon2[0].shape[1]
NB_f1 = 38

# downsample the second dataset
timeSeriesCon2_new = np.empty(len(label2),dtype=object)
sizes2_2_new = np.zeros(len(label2))
for i in range(len(label2)):
    temp = timeSeriesCon2[i]
    indices = np.flip(np.arange(temp.shape[0]-1,-1,-2),axis=0)
    timeSeriesCon2_new[i] = temp[indices,:]
    sizes2_2_new[i] = indices.shape[0]
    

# remove std=0 features in the first dataset: [14,16,26]
remove_f = np.arange(10,28)
indices = np.array(list(set(np.arange(NB_f1))-set(remove_f)))
timeSeriesCon_new = np.empty(len(label),dtype=object)
means1,_ = compute_std(50,1,timeSeriesCon)
std1_r = 1
means1 = means1[indices]
low = np.expand_dims(means1-2*std1_r, 0)
high = np.expand_dims(means1+2*std1_r, 0)
for i in range(len(label)):
    temp = timeSeriesCon[i]
    temp_dyna = temp[:,indices]
    temp_sta = temp[:,NB_f1:]
    temp_dyna = np.clip(temp_dyna, low, high)
    timeSeriesCon_new[i] = np.concatenate((temp_dyna,temp_sta),axis=1)
dim = timeSeriesCon_new[0].shape[1]
NB_f1 = indices.shape[0]
_,std1 = compute_std(50,1,timeSeriesCon)
_,std1 = compute_std(50,1,timeSeriesCon_new)
plt.plot(std1)

# split the dataset
timeSeriesCon_new0 = timeSeriesCon_new[label==0]
timeSeriesCon_new1 = timeSeriesCon_new[label==1]
timeSeriesCon2_new0 = timeSeriesCon2_new[label2==0]
timeSeriesCon2_new1 = timeSeriesCon2_new[label2==1]




# method 1
def estimated_correlation(x,y):
    n = x.shape[0]
    x = x-x.mean()
    y = y-y.mean()
    r = np.correlate(x, y, mode = 'full')[-n:]
    #assert np.allclose(r, np.array([(y[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/np.arange(n, 0, -1)
    return result

@pu.memoize('./newres/new_correlation_{leng:d}_{dataset:d}_ave.pkl.gz', log_level='warn')
def corr_compute(leng,dataset):
    nb_f = NB_f1 if dataset==1 else 40
    stream = timeSeriesCon_new if dataset==1 else timeSeriesCon2_new
    sizes = sizes2 if dataset==1 else sizes2_2_new
    l = label if dataset==1 else label2
    indices = np.nonzero(sizes>=leng)[0]
    corr = np.zeros((indices.shape[0],nb_f,nb_f,leng))
    for i in range(indices.shape[0]):
        for j in range(nb_f):
            for k in range(nb_f):
                temp1 = stream[indices[i]][-leng:,j]
                temp2 = stream[indices[i]][-leng:,k]
                corr[i,j,k,:] =  estimated_correlation(temp1,temp2)
    return np.mean(corr,axis=0)

cube1 = corr_compute(leng=50,dataset=1)
cube2 = corr_compute(leng=50,dataset=2)

cube1 = pu.load("./newres/new_correlation_50_1_ave.pkl.gz")
cube2 = pu.load("./newres/new_correlation_50_2_ave.pkl.gz")

def compute_std(leng,dataset,timeSeries=timeSeriesCon):
    nb_f = NB_f1 if dataset==1 else 40
    stream = timeSeries if dataset==1 else timeSeriesCon2_new
    sizes = sizes2 if dataset==1 else sizes2_2_new
    l = label if dataset==1 else label2
    indices = np.nonzero(sizes>=leng)[0]
    corr = np.zeros((indices.shape[0],nb_f,nb_f,leng))
    means = np.zeros(nb_f)
    stds = np.zeros(nb_f)
    N = leng*indices.shape[0]
    for i in range(indices.shape[0]):
        means += np.sum(stream[indices[i]][-leng:,:nb_f],axis=0)
    means = means/N
    for i in range(indices.shape[0]):
        stds += np.sum(np.power(stream[indices[i]][-leng:,:nb_f]-means,2),axis=0)
    stds = np.sqrt(stds/(N-1))
    return means, stds

_,std1 = compute_std(50,1,timeSeriesCon_new)  
_,std2 = compute_std(50,2)  

def divide_std(std, cube):
    temp1 = np.expand_dims(std,0)
    temp2 = np.expand_dims(std,1)
    temp = np.expand_dims(np.matmul(temp2,temp1),axis=2)
    return np.divide(cube,temp)

cube1_c = divide_std(std1,cube1)
cube2_c = divide_std(std2,cube2)

corr_norm2 = np.linalg.norm(cube2_c,ord='fro',axis=(0,1))/(cube2_c.shape[0]*cube2_c.shape[1])
corr_norm1 = np.trace(np.abs(cube1_c),axis1=0, axis2=1)/(cube1_c.shape[0])
corr_norm2 = np.trace(np.abs(cube2_c),axis1=0, axis2=1)/(cube2_c.shape[0])

ave1 = np.zeros_like(cube1_c[i,i,:])
for i in top5_1:#range(NB_f1):
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.plot(cube1_c[i,i,:])
    ave1 += cube1_c[i,i,:]
    ax.set_title("ICU, feature {}".format(i))
ave1 /= len(top5_1)
ave2 = np.zeros_like(cube2_c[i,i,:])
for i in top5_2:#range(40):
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.plot(cube2_c[i,i,:])
    ave2 += cube2_c[i,i,:]
    ax.set_title("MIMIC-III, feature {}".format(i))
ave2 /= len(top5_2)


#
f = plt.figure()
ax = f.add_subplot(211)
axi = np.arange(corr_norm1.shape[0])*2
ax.plot(axi,corr_norm1,label="(Alaa et al. 2017) dataset")
ax.plot(axi,corr_norm2,label="MIMIC-III dataset")
ax.set_xlabel("Time Lag (hours)")
ax.set_ylabel("Autocorrelation (absolute values)")
ax.legend(loc=0)
ax.set_title("Average autocorrelation of time-series features")
savemat("fig_autocorr",{"corr_norm1":corr_norm1,"corr_norm2":corr_norm2,"axi1":axi})



# method 2: Hurst Exponent
def hurst(ts):
    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)
    # Calculate the array of the variances of the lagged differences
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)
    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0

# dataset 1: ICU
hurst1 = []
for i in range(len(label)):
    temp = []
    for j in range(NB_f1):
        temp.append(hurst(timeSeriesCon[i][:,j]))
    hurst1.append(temp)

hurst1 = np.array(hurst1)
nb_nan = []
for i in range(hurst1.shape[1]):
    nb_nan.append(np.sum(np.isnan(hurst1[:,i])))
nb_nan = np.array(nb_nan)

pu.dump(hurst1,"hurst1.pkl")
        
mean_hurst1 = np.divide(np.sum(np.abs(np.nan_to_num(hurst1-0.5)),axis=0),nb_nan)

# dataset 2:MIMIC
hurst2 = []
for i in range(len(label2)):
    temp = []
    for j in range(40):
        temp.append(hurst(timeSeriesCon2[i][:,j]))
    hurst2.append(temp)
hurst2 = np.array(hurst2)

nb_nan2 = []
for i in range(hurst2.shape[1]):
    nb_nan2.append(np.sum(np.isnan(hurst2[:,i])))
nb_nan2 = np.array(nb_nan2)

pu.dump(hurst2,"hurst2.pkl")
        
mean_hurst2 = np.divide(np.sum(np.abs(np.nan_to_num(hurst2-0.5)),axis=0),nb_nan2)



# method 3: Correlation with labels
def corr_compute(leng,dataset):
    nb_f = NB_f1 if dataset==1 else 40
    stream = timeSeriesCon if dataset==1 else timeSeriesCon2_new
    sizes = sizes2 if dataset==1 else sizes2_2_new
    l = label if dataset==1 else label2
    indices = np.nonzero(sizes>=leng)[0]
    features = np.zeros((nb_f,indices.shape[0],leng))
    for i in range(indices.shape[0]):
        for j in range(nb_f):
            features[j,i,:] = stream[indices[i]][-leng:,j]

    corr = np.zeros((nb_f,features.shape[2]))
    for i in range(features.shape[0]):
        for j in range(features.shape[2]):
            corr[i,j],_ = pearsonr(features[i,:,j],l[indices])
    return corr


corr1 = corr_compute(50,1)
corr2 = corr_compute(50,2)

def plot_corr(corr,leng,threshold=0):
    time_axis = -np.arange(leng-1,-1,-1)*2
    for i in range(corr.shape[0]):
        # check if there is nan
        if(np.sum(np.isnan(corr[i,:]))==0 and np.abs(corr[i,:]).max()>=threshold):
            f = plt.figure()
            ax = f.add_subplot(111)
            ax.plot(time_axis,np.abs(corr[i,:]))
            ax.set_xlabel("Time Proceding Label (hours)")
            ax.set_ylabel("Pearson Correlation Coefficient (absolute value)")
            name = "Evolution of correlation between feature {} and label".format(i)
            ax.set_title(name)
            
def plot_corr_average(leng,threshold=0,window_size=51,order=3):
    time_axis = -np.arange(leng-1,-1,-1)*2
    indices1 = []
    indices2 = []
    for i in range(corr1.shape[0]):
        # check if there is nan
        if(np.sum(np.isnan(corr1[i,:]))==0 and np.abs(corr1[i,:]).max()>=threshold):
            indices1.append(i)
    for i in range(corr2.shape[0]):
        # check if there is nan
        if(np.sum(np.isnan(corr2[i,:]))==0 and np.abs(corr2[i,:]).max()>=threshold):
            indices2.append(i)
    indices1 = np.array(indices1)
    indices2 = np.array(indices2)
    ax = f.add_subplot(212)
    curve1 = np.mean(np.abs(corr1[indices1,:]),axis=0)
    curve1_sm = savitzky_golay(curve1,window_size=window_size,order=order)
    #ax.plot(time_axis,curve1,label="ICU")
    ax.plot(time_axis,savitzky_golay(curve1,window_size=window_size,order=order),label="(Alaa et al. 2017) dataset")
    curve2 = np.mean(np.abs(corr2[indices2,:]),axis=0)
    curve2_sm = savitzky_golay(curve2,window_size=window_size,order=order)
    #ax.plot(time_axis,curve2,label="mortality")
    ax.plot(time_axis,savitzky_golay(curve2,window_size=window_size,order=order),label="MIMIC-III dataset")
    ax.set_xlabel("Time Proceding Label (hours)")
    ax.set_ylabel("Correlation (absolute value)")
    name = "Pearson correlation between time-series features and labels"
    ax.set_title(name)
    ax.legend(loc=0)
    
    
savemat("fig_corrlabel",{"curve1_sm":curve1_sm,"curve2_sm":curve2_sm,"axi2":time_axis})
    
plot_corr_average(50,0.2,23,3)
plot_corr(corr1,50,0.2)
plot_corr(corr2,50,0.2)



# find the top 5 predictive features
sorted1 = np.argsort(np.max(corr1,axis=1))
top5_1 = sorted1[-5:]
sorted2 = np.argsort(np.max(corr2,axis=1))
top5_2 = sorted2[-5:]
