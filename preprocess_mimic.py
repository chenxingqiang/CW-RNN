'''
Preprocessing MIMIC-III dataset (MIMIC-III, a freely accessible critical care database. Johnson AEW, Pollard TJ, Shen L, Lehman L, Feng M, Ghassemi M, Moody B, Szolovits P, Celi LA, and Mark RG. Scientific Data (2016).). We extract 40 least missing time-series features
'''
import numpy as np
import pandas as pd
import itertools as it
import pickle_utils as pu
import tensorflow as tf
import gzip
import csv
import math
import os
import os.path
from tqdm import tqdm
from scipy.interpolate import pchip
import math


MIMIC_PATH = '/Users/xuan/Desktop/oxford/project/MIMIC/CSV/mimic.csv.gz'
STATIC_PATH = '/Users/xuan/Desktop/oxford/project/MIMIC/CSV/static_patients.csv.gz'
CSV_PATH = '/Users/xuan/Desktop/oxford/project/MIMIC/CSV/'


def get_headers(table):
    "Get the headers of a MIMIC sub-table"
    with gzip.open(CSV_PATH+'{:s}.csv.gz'.format(table), 'rt',
            newline='') as csvf:
        return next(iter(csv.reader(csvf)))[3:]
        

@pu.memoize('dataset/parsed_csv.pkl.gz', log_level='warn')
def parse_csv(usecols, dtype=None):
    return pd.read_csv(MIMIC_PATH, header=0, index_col='icustay_id',
                       usecols=usecols, dtype=dtype, engine='c',
                       true_values=[b'1'], false_values=[b'0', b''])
    
    
@pu.memoize('dataset/number_non_missing.pkl.gz', log_level='warn')
def number_non_missing():
    with gzip.open(MIMIC_PATH, 'rt') as gzf:
        f = csv.reader(gzf)
        headers = next(f)
        non_missing = np.zeros(len(headers), dtype=np.int32)
        n_lines = 0
        for line in tqdm(f):
            n_lines += 1
            for i, e in enumerate(line):
                if e!='':
                    non_missing[i] += 1
    n_missing_headers = list(zip(non_missing, headers))
    return n_missing_headers, n_lines


@pu.memoize('dataset/useful_vs_headers.pkl.gz', log_level='warn')
def get_recommanded_vitalsigns_headers():
    mimic_headers = get_headers('mimic')
    df = pd.read_csv('~/Desktop/oxford/project/MIMIC/CSV/vital_signs_headers.csv')
    names = df.iloc[:,3].dropna(axis=0, how='any').values
    for (i,n) in enumerate(names):
        names[i] = n.replace('.',' ')
    output = []
    for e in names:
        if e in mimic_headers:
            output.append(e)
    for i in [956, 1012, 2017]:
        output.append(mimic_headers[i])
        # F Peak Insp (Pressure), F Richmond RAS Scale, F Pupils, can't find 'F Side Rails', 'F Pupil Size Right', 'F Pupil Size Left',
    return output


def get_realvalue_headers():
    mimic_headers = get_headers('mimic')
    realvalue_headers = filter(lambda h: h[0]=='F', mimic_headers)
    return realvalue_headers


@pu.memoize('dataset/ts_headers_{n_frequent:d}.pkl.gz', log_level='warn')
def get_frequent_headers(n_frequent):
    "Parses the MIMIC csv, grabbing only the `n_frequent` most frequent headers"
    lab_headers = get_headers('labevents')
    r_headers = get_realvalue_headers()
    vs_headers = list(set(r_headers)-set(lab_headers))

    count_headers, _ = number_non_missing()
    count_headers.sort()
    _, headers = zip(*count_headers)
    data_headers = filter(lambda h: (h in lab_headers) or (h in vs_headers), headers)
    data_headers = list(data_headers)[-n_frequent:]
    
    return data_headers


@pu.memoize('dataset/static_data.pkl.gz', log_level='warn')
def get_static_data():
    numerical = ["r_admit_time", "b_gender", "r_age", "i_previous_admissions",
               "i_previous_icustays"]
    categorical = ["c_admit_type", "c_admit_location", "c_ethnicity"]
    label = ['b_pred_died_in_hospital']
    label_time = ['r_pred_death_time', 'r_pred_discharge_time']
    usecols = ["icustay_id"] + numerical + categorical + label + label_time

    dtype = dict(it.chain(
        zip(numerical, it.repeat(np.float32)),
        zip(categorical, it.repeat('category'))))
    df = pd.read_csv(STATIC_PATH, header=0, index_col="icustay_id",
                     usecols=usecols, dtype=dtype, engine='c',
                     true_values=[b'1'], false_values=[b'0', b''])
    index = df.index
    df.r_admit_time = df.r_admit_time.apply(lambda n: n/3600)
    df.r_pred_death_time = df.r_pred_death_time.apply(lambda n: n/3600)
    df.r_pred_discharge_time = df.r_pred_discharge_time.apply(lambda n: n/3600)
    _true_label_time = []
    for i in range(df.shape[0]):
        if(df.loc[index[i], label[0]]==0):
            _true_label_time.append(df.loc[index[i], label_time[1]])
        else:
            _true_label_time.append(df.loc[index[i], label_time[0]])
    df = df.assign(true_label_time=np.array(_true_label_time))
    df[categorical] = df[categorical].applymap(int)
    return df[numerical+categorical], df[label], df['true_label_time']


@pu.memoize('dataset/data_{n_frequent:d}.pkl.gz', log_level='warn')
def get_data(n_frequent):
    ts_headers = get_frequent_headers(n_frequent=n_frequent)
    sta_data, la, la_time = get_static_data()
    mimic = parse_csv(['icustay_id', 'hour'] + ts_headers + ['F pred hours_until_death'], dtype=None)
    
    mimic['F pred hours_until_death'] = mimic['F pred hours_until_death'].apply(lambda n: n/3600)
    ts = []
    timestamp = []
    label = []
    label_time = []
    static_data = []
    icustay_ids = []
    hours_until_death = []
    for icustay_id, df in mimic.groupby(level=0):
        print("Doing icustay_id", icustay_id, "...")
        if(df['hour'].values.size==0 or df['hour'].values[-1]<0):
            continue
        ts.append(df[ts_headers].values)
        timestamp.append(df['hour'].values)
        hours_until_death.append(df['F pred hours_until_death'].values)
        
        static_data.append(sta_data.loc[icustay_id].values)
        label.append(la.loc[icustay_id].values)
        label_time.append(la_time.loc[icustay_id])
        
        icustay_ids.append(icustay_id)
        
        
    # should output the list of icustay for future reference
    return ts, timestamp, static_data, label, label_time, icustay_ids, hours_until_death
   

def zero_hold_interpolate_1d(ts_1d, length):
    new_ts_1d = np.zeros(length)
    detect_nan = np.vectorize(lambda x: math.isnan(x))
    num_indices = np.nonzero(detect_nan(ts_1d)==False)[0]
    # compute missing mask
    num_missing = np.nonzero(detect_nan(ts_1d)==True)[0]
    missing_mask = np.zeros_like(new_ts_1d)
    missing_mask[num_missing] = 1
    # if ts_1d is empty
    if (num_indices.size==0):
        return new_ts_1d, missing_mask
    # interpolate
    if num_indices[0]>0:
        new_ts_1d[:num_indices[0]] = ts_1d[num_indices[0]] # begin
    for i in range(num_indices.size-1): # middle
        start = num_indices[i]
        end = num_indices[i+1]
        new_ts_1d[start:end] = ts_1d[num_indices[i]]
    new_ts_1d[(num_indices[-1]+1):] = ts_1d[num_indices[-1]] # end
    return new_ts_1d, missing_mask
 
    
def zero_hold_interpolate(ts, timestamp):
    # delete time series before ICU
    if(timestamp[0]<0):
        delete_in = np.nonzero(timestamp<0)[0][-1]+1
        ts = ts[delete_in:,:]
        timestamp = timestamp[delete_in:]
    # if timestamp is not regular
    length = timestamp[-1] + 1
    if(timestamp.size<length): 
        temp = np.zeros([length, ts.shape[1]])
        indices = list(set(np.arange(length))-set(timestamp))
        temp[timestamp,:] = ts
        temp[indices,:] = math.nan
        ts = temp
    # zero_hold interpolate
    new_ts = np.zeros([length, ts.shape[1]])
    missing_mask = np.zeros_like(new_ts)
    for i in range(ts.shape[1]):
        new_ts[:,i], missing_mask[:,i] = zero_hold_interpolate_1d(ts[:,i], length)
    new_timestamp = np.arange(length)
    return new_ts, new_timestamp, missing_mask
        

def bad_interpolate(ts, timestamp):
    if(timestamp[0]<0):
        delete_in = np.nonzero(timestamp<0)[0][-1]+1
        ts = ts[delete_in:,:]
        timestamp = timestamp[delete_in:]
    detect_nan = np.vectorize(lambda x: math.isnan(x))
    missing_number = np.sum(detect_nan(ts), axis=0)
    ratio = 0.9
    cols_miss_many = np.nonzero(missing_number>=math.floor(ratio*ts.shape[1]))[0]
    cols_miss = np.array(list(set(np.arange(ts.shape[1])) - set(cols_miss_many)))
    
    length = timestamp[-1] + 1
    new_ts = np.zeros([length, ts.shape[1]])
    for i in cols_miss:
        num_indices = np.nonzero(detect_nan(ts[:,i])==False)[0]
        nan_indices = list(set(np.arange(length))-set(num_indices))
        interp = pchip(timestamp[num_indices], ts[num_indices,i])
        new_ts[num_indices, i] = ts[num_indices, i]
        new_ts[nan_indices, i] = interp(nan_indices)
        if(np.sum(detect_nan(new_ts[:,i]))>1):
            print('1,',i)
    for i in cols_miss_many:
        num_indices = np.nonzero(detect_nan(ts[:,i])==False)[0]
        nan_indices = list(set(np.arange(length))-set(num_indices))
        mean_value = np.mean(ts[num_indices,i]) if num_indices.size>0 else 0
        new_ts[nan_indices, i] = mean_value
        if(np.sum(detect_nan(new_ts[:,i]))>1):
            print('2,',i)
    
    return new_ts


@pu.memoize('dataset/interpolated_ts_mask_{method:s}.pkl.gz', log_level='warn')
def interpolate(ts, timestamp, method='zero_hold'):
    if(method=='zero_hold'):
        interp_fun = zero_hold_interpolate
    new_ts = []
    new_timestamp = []
    missing_mask = []
    for i in range(len(ts)):
        print("Interpolating ", i, "...")
        temp = interp_fun(ts[i], timestamp[i])
        new_ts.append(temp[0])
        new_timestamp.append(temp[1])
        missing_mask.append(temp[2])
    return new_ts, new_timestamp, missing_mask


# subtract mean, divide standard deviation
def normalize(stream):
    N = stream.shape[0]
    sums = np.zeros(stream[0].shape[1])
    sizes = 0
    for i in range(N):
        sums += np.sum(stream[i], axis=0)
        sizes += stream[i].shape[0]
    means = np.divide(sums, sizes)
    sums = np.zeros(stream[0].shape[1])
    for i in range(N):
        sums += np.sum(np.power(stream[i]-means,2), axis=0)
    std = np.sqrt(np.divide(sums, sizes-1))
    normalized_stream = np.empty(N, dtype=stream.dtype)
    for i in range(N):
        normalized_stream[i] = np.divide(stream[i]-means, std)
    return normalized_stream
    

def conFeatures(indices, stream, missing_mask, static, sizes, cut=False):
    N = indices.size
    output = np.empty(N,dtype=object)
    for i in range(N):
        i1 = indices[i]
        if(cut):
            output[i] = np.hstack([stream[i1][:,-7:], missing_mask[i1][:,-7:], \
                                  np.matlib.repmat(static[i1], sizes[i1], 1)])
        else:
            output[i] = np.hstack([stream[i1], missing_mask[i1], \
                              np.matlib.repmat(static[i1], sizes[i1], 1)])
    return output

def conFeaturesNoMask(indices, stream, static, sizes, cut=False):
    N = indices.size
    output = np.empty(N,dtype=object)
    for i in range(N):
        i1 = indices[i]
        if(cut):
            output[i] = np.hstack([stream[i1][:,-7:],  \
                                  np.matlib.repmat(static[i1], sizes[i1], 1)])
        else:
            output[i] = np.hstack([stream[i1],  \
                              np.matlib.repmat(static[i1], sizes[i1], 1)])
    return output


if __name__ == '__main__':
    ts, timestamp, static_data, label, label_time, icustay_ids, \
        hours_until_death = get_data(n_frequent=40)
    new_ts, new_timestamp, missing_mask = interpolate(ts, timestamp, method='zero_hold')
    (new_ts, new_timestamp, missing_mask, static_data, label, icustay_ids) = map(lambda e:
        np.squeeze(np.array(e)), (new_ts, new_timestamp, missing_mask, static_data, label, icustay_ids)
        )
    
    # normalisation of data
    new_ts = normalize(new_ts)
    # only use time series not too short and not too long
    sizes = np.array([x.size for x in new_timestamp])
    indices_save = np.nonzero(np.logical_and(sizes>10,sizes<1500))[0]
    # inverse missing mask
    missing_mask = 1 - missing_mask
    timeSeriesCon = conFeaturesNoMask(indices_save, new_ts, static_data, sizes)
    pu.dump((timeSeriesCon, label[indices_save], sizes[indices_save], 
             icustay_ids[indices_save]), 'dataset/data_nonorm.pkl.gz')
    

    
