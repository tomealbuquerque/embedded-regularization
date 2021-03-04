import numpy as np 
import pickle 
import _pickle as cPickle

def dump_to_pickle(obj, filename_dir, mode="pickle"):

    if mode=="pickle":
        with open(filename_dir, 'wb') as f:
            pickle.dump(obj, f, -1)
    
    elif mode=="cPickle":
        with open(filename_dir, 'wb') as f:
            cPickle.dump(obj, f, -1)
    
    return None


def open_from_pickle(filename_dir, mode="pickle"):

    if mode=="pickle":
        with open(filename_dir, 'rb') as f:
            obj = pickle.load(f)

    elif mode=="cPickle":
        with open(filename_dir, 'rb') as f:
            obj = cPickle.load(f)
    
    return obj
