from __future__ import print_function
import os
import numpy as np
import fitsio
from astrometry.util.util import *
from astrometry.util.file import *
from astrometry.util.plotutils import *
import pylab as plt

'''
Try 'fixing' the Lanczos kernel by iteratively computing the centroid
of the kernel and offsetting by that amount.
'''
ps = PlotSequence('lan')

subsampling = 100

xx = np.arange(-4., 3+1.).astype(np.float32)

lanczos_kernel = np.zeros((len(xx) * subsampling))
fixed_kernel = np.zeros((len(xx) * subsampling))
fixed2_kernel = np.zeros((len(xx) * subsampling))

fixed_x = np.zeros(subsampling)
fixed_x2 = np.zeros(subsampling)

cen0 = np.zeros(subsampling)
cen1 = np.zeros(subsampling)
cen2 = np.zeros(subsampling)

DX = np.linspace(0, 1, subsampling, endpoint=False)
for i,dx in enumerate(DX):
    L = np.zeros_like(xx)
    rtn = lanczos3_filter(xx + dx, L)
    assert(rtn == 0)
    lanczos_kernel[i::subsampling] = L

    print('dx', dx)
    Lsum = L.sum()
    #print('sum', L.sum())
    centroid = -np.sum(L * xx) / Lsum
    print('centroid', centroid, 'vs', dx)
    cen0[i] = centroid

    fx = dx - (centroid - dx)
    rtn = lanczos3_filter(xx + fx, L)
    assert(rtn == 0)
    fixed_kernel[i::subsampling] = L
    #print('sum', L.sum())
    centroid = -np.sum(L * xx) / Lsum
    print('centroid', centroid, 'vs', dx)
    cen1[i] = centroid
    fixed_x[i] = fx
    
    fx2 = fx - (centroid - dx)
    rtn = lanczos3_filter(xx + fx2, L)
    assert(rtn == 0)
    fixed2_kernel[i::subsampling] = L
    fixed_x2[i] = fx2
    #print('sum', L.sum())
    centroid = -np.sum(L * xx) / Lsum
    print('centroid', centroid, 'vs', dx)
    cen2[i] = centroid
    
plt.clf()
plt.plot(lanczos_kernel, 'b-')
plt.plot(fixed_kernel, 'r-')
plt.plot(fixed2_kernel, 'g-')
ps.savefig()
    
plt.clf()
plt.plot(DX, cen0, 'b-')
plt.plot(DX, cen1, 'r-')
plt.plot(DX, cen2, 'g-')
ps.savefig()

plt.clf()
plt.plot(DX, cen0 - DX, 'b-')
plt.plot(DX, cen1 - DX, 'r-')
plt.plot(DX, cen2 - DX, 'g-')
ps.savefig()
