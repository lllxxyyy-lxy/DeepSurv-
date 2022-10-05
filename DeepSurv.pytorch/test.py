import h5py
import os
import numpy as np
import deepdish as dd
import pandas as pd

# path = "data/metabric/metabric_IHC4_clinical_train_test.h5"

# f = h5py.File(path,'r')  

# for group in f.keys():
#     group_read = f[group]
#     print(list(group_read.keys()))
#     for subgroup in group_read.keys():
#         set1 = f[group + '/' + subgroup]
#         print (set1.name)
#         data = np.array(set1)
#         print (data.shape)
#         #print(data)
    
# a = np.array(f['train/e'])  
# np.savetxt('e.csv', a, delimiter = ",")

data = pd.read_table('mirna', sep = ',',skiprows=0)
data = pd.DataFrame(data.values.T, index=data.columns, columns=data.index)
data = data.to_csv('mRNA.csv',header=False, index=False)