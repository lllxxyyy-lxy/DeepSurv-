
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

class SurvivalDataset(Dataset):
    
    def __init__(self, h5_file, is_train):

        self.X, self.e, self.y = \
            self._read_h5_file(h5_file, is_train)

        self._normalize()

        print('=> load {} samples'.format(self.X.shape[0]))

    def _read_h5_file(self, h5_file, is_train):

        split = 'train' if is_train else 'test'
        with h5py.File(h5_file, 'r') as f:
            X = f[split]['x'][()]
            e = f[split]['e'][()].reshape(-1, 1)
            y = f[split]['t'][()].reshape(-1, 1)
        return X, e, y

    def _normalize(self):
       
        self.X = (self.X-self.X.min(axis=0)) / \
            (self.X.max(axis=0)-self.X.min(axis=0))

    def __getitem__(self, item):
      
        X_item = self.X[item] 
        e_item = self.e[item] 
        y_item = self.y[item] 

        X_tensor = torch.from_numpy(X_item)
        e_tensor = torch.from_numpy(e_item)
        y_tensor = torch.from_numpy(y_item)
        return X_tensor, y_tensor, e_tensor

    def __len__(self):
        return self.X.shape[0]