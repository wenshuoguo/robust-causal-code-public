import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

import pickle
import os

AA = np.zeros([5,3])
for noise in [1,2,3,4,5]:
    RR = []
    SS = []
    for seed in [1,2,3,4]:
        A2 = pickle.load(open('./clean_out/clean','rb'))
        
        A = pickle.load(open('./boot_out/noise'+str(int(noise))+'_seedn'+str(int(seed)),'rb'))
        
        R = np.zeros([12,4])
        R[:] = np.NaN
        R[:,0] = np.arange(1,13)
        
        for j in range(len(R)):
            R[j,0]
            for a in A2:
                if a[0]==R[j,0]:
                    R[j,1]=a[1]
            for a in A:
                if a[0]==R[j,0]:
                    R[j,2], R[j,3] = np.percentile(a[1],0),np.percentile(a[1],95)
        
        R1 = R[~np.isnan(R).any(axis=1), :]
        RR.append(R1)
        
        ct = 0
        for a in R1:
            if a[2]<a[1]<a[3]:
                ct = ct+1
        SS.append(ct/len(R1))
        
    RR = np.vstack(RR)
    ct = 0
    for a in RR:
        if a[2]<a[1]<a[3]:
            ct = ct+1
    print('######')
    print('noise', noise, ct/len(RR), np.std(SS))
    AA[noise-1,:] = [noise, ct/len(RR), np.std(SS)]

print(pd.DataFrame(AA).round(2).to_latex(index=False))

    