# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:14:31 2020

@author: albu
"""
import pandas as pd
from skimage.io import imread , imshow
import os
import numpy as np
import pickle

d=pd.read_pickle('nci_dados.pkl')

n=pd.DataFrame(d).to_numpy()

def load_image(i, imgname):
    #print('start')
    # i, imgname = args
    if i % 100 == 0:
        print(i)
    img = imread(os.path.join('preprocess_imgs', imgname))
    print(img.shape)
    # img = resize(img, SIZE, mode='constant', anti_aliasing=True)
    # img = (img * 255).astype(np.uint8)
    #print('end')
    return img

data=[]
for idx, imgname in enumerate(d['image']):
    print(idx)
    data.append(load_image(idx,imgname))
    
print(np.shape(d))

# d['image'] = [e for e in data]
# n=pd.DataFrame(d).to_numpy()
# d.to_pickle("nci_teste.pkl")
# pickle.dump(n, open('nci_teste.pickle', 'wb'))
