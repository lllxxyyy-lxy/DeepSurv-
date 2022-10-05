
from ast import comprehension
from unicodedata import name
import h5py
import os
import numpy as np
import deepdish as dd
import pandas as pd
import numpy as np

train_e = np.loadtxt(open('test_data/train_e.csv',"rb"), delimiter=",", skiprows=0)
train_t = np.loadtxt(open('test_data/train_t.csv',"rb"), delimiter=",", skiprows=0)
train_x = np.loadtxt(open('test_data/train_x.csv',"rb"), delimiter=",", skiprows=0)

test_e = np.loadtxt(open('test_data/test_e.csv',"rb"), delimiter=",", skiprows=0)
test_t = np.loadtxt(open('test_data/test_t.csv',"rb"), delimiter=",", skiprows=0)
test_x = np.loadtxt(open('test_data/test_x.csv',"rb"), delimiter=",", skiprows=0)

with h5py.File('brain_cancer_train_test.h5', "w") as f:
   grp1 = f.create_group('train')
   grp1.create_dataset(name = "e", data = train_e, dtype='f')
   grp1.create_dataset(name = "t", data = train_t, dtype='f')
   grp1.create_dataset(name = "x", data = train_x, dtype='f')
   
   grp2 = f.create_group('test')
   grp2.create_dataset(name = "e", data = test_e, dtype='f')
   grp2.create_dataset(name = "t", data = test_t, dtype='f')
   grp2.create_dataset(name = "x", data = test_x, dtype='f')


   
    

    