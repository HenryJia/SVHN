import h5py
import scipy.io as sio
import numpy as np

def get_labels(fn):
    raw = sio.loadmat(fn)['digitStruct']
    targets = []
    for i in xrange(raw.shape[1]):
        label = []
        label_raw = raw[0, i][1]
        for j in xrange(label_raw.shape[1]):
            label += [label_raw[0, j][-1][0, 0]]
        targets += [label]
    return targets
