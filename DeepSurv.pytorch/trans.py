from asyncore import file_dispatcher
import numpy as np
import sys
import h5py
path = "data/metabric/metabric_IHC4_clinical_train_test.h5"
file = h5py.File(path,'r')

np.savetxt(sys.stdout, h5py.File(path)['dataname'], '%g', ',')
