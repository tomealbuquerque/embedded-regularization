# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:14:27 2020

*** CODE FOR DATAFRAME BUILD of NCI DATA ***

@author: Tomé Albuquerque
"""

import pickle
from tensorflow.keras import utils
from skimage import draw
import pickle
import numpy as np
import pandas as pd
import os

#import excel files with data
folders = [
    'ALTS',
    'Biopsy_Study',
    'CVT',
    'NHS',
]

def load_excel(folder):
    filename = os.path.join('covariate_data_training_'+folder+'.xls')
    df = pd.read_excel(filename, skiprows=2, na_values='.')
    df['PATIENT_ID'] = ['%s_%s' % (folder[:3], id) for id in df['PATIENT_ID']]
    df['IMAGE_ID'] = [os.path.join(id) for id in df['IMAGE_ID']]
    df = df.rename(columns={'IMAGE_ID': 'image', 'PATIENT_ID': 'patient'})
    df.columns = map(str.lower, df.columns)
    return df

dfs = [load_excel(folder) for folder in folders]
d = pd.concat(dfs)

   
# aproveitar teste do HPV para inferir o resultado do Hist (transformar em 2 classes)
for i in range(len(d)):
    print(i)
    if d.iloc[i]['wrst_hist_after'] ==-2:
        if d.iloc[i]['hpv_status'] ==0 or d.iloc[i]['hpv_status'] ==1:
            d.iloc[i,3]=0
        elif d.iloc[i]['hpv_status'] ==2 or d.iloc[i]['hpv_status'] ==3:
            d.iloc[i,3]=1
    elif d.iloc[i]['wrst_hist_after'] ==2 or d.iloc[i]['wrst_hist_after'] ==3 or d.iloc[i]['wrst_hist_after'] ==4:
        d.iloc[i,3]=1
    else:
        d.iloc[i,3]=0

for i in range(len(d)):
    if np.isnan(d.iloc[i]['wrst_hist_after_dt'])==True:
        d.iloc[i,4]=-10
        print(i)
unk=[]
for i in range(len(d)):
    if d.iloc[i]['wrst_hist_after'] ==-2:
        print(i)
        unk.append(i)
        
d=d.drop(d.index[unk])      

agetr = d['age_grp'][:, np.newaxis] / np.amax(d['age_grp'][:, np.newaxis])
    
hpvtr = d['hpv_status'][:, np.newaxis] / np.amax(d['hpv_status'][:, np.newaxis])
   
timetr = d['timepnt'][:, np.newaxis] / np.amax(d['timepnt'][:, np.newaxis])
   
ydttr = d['wrst_hist_after_dt'][:, np.newaxis] / np.amax(d['wrst_hist_after_dt'][:, np.newaxis])

Clintr = np.concatenate((agetr, hpvtr, timetr, ydttr), axis=-1)

d['age']=Clintr[:,0]
d['hpv']=Clintr[:,1]
d['time']=Clintr[:,2]
d['ydt']=Clintr[:,3]

dataframe = d.drop(['patient', 'hpv_dt','age_grp', 'wrst_hist_after_dt', 'hpv_status', 'timepnt'], axis=1)
dataframe=dataframe.rename(columns={"wrst_hist_after": "label"})
dataframe.to_pickle("nci_dados.pkl")
