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

subsampling = 256

xx = np.arange(-4., 3+1.).astype(np.float32)
xx += 0.5 * (1./subsampling)

lanczos_kernel = np.zeros((len(xx) * subsampling))
fixed_kernel = np.zeros((len(xx) * subsampling))
fixed2_kernel = np.zeros((len(xx) * subsampling))

subx = np.zeros((len(xx) * subsampling))

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

    subx[i::subsampling] = xx + dx
    
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

print('subx', subx.min(), subx.max())
    
plt.clf()
plt.plot(subx, lanczos_kernel, 'b-', lw=3, alpha=0.5, label='Lanczos')
#plt.plot(fixed_kernel, 'r-')
plt.plot(subx, fixed2_kernel, 'r-', label='"fixed(2)"')
plt.xlim(-3.5, 3.5)
plt.legend()
ps.savefig()
    
plt.clf()
plt.plot(DX, cen0, 'b-')
plt.plot(DX, cen1, 'g-')
plt.plot(DX, cen2, 'r-')
ps.savefig()

plt.clf()
plt.plot(DX, cen0 - DX, 'b-', label='Lanczos-3')
plt.plot(DX, cen1 - DX, 'g-', label='"fixed(1)"')
plt.plot(DX, cen2 - DX, 'r-', label='"fixed(2)"')
plt.xlabel('subpixel offset (pixels)')
plt.ylabel('centroid error (pixels)')
plt.legend()
ps.savefig()

plt.clf()
plt.plot(DX, fixed_x - DX, 'g-')
plt.plot(DX, fixed_x2 - DX, 'r-')
ps.savefig()

F0 = np.fft.rfft(lanczos_kernel)
F1 = np.fft.rfft(fixed_kernel)
F2 = np.fft.rfft(fixed2_kernel)

print(len(lanczos_kernel))

plt.clf()
plt.subplot(2,1,1)
plt.plot(F0.real, 'b-')
plt.plot(F1.real, 'g-')
plt.plot(F2.real, 'r-')
plt.xscale('log')
plt.ylabel('Fourier transform (real)')
plt.subplot(2,1,2)
plt.plot(F0.imag, 'b-')
plt.plot(F1.imag, 'g-')
plt.plot(F2.imag, 'r-')
plt.xscale('log')
plt.ylabel('Fourier transform (imag)')
ps.savefig()

mag0 = np.hypot(F0.real, F0.imag)
mag1 = np.hypot(F1.real, F1.imag)
mag2 = np.hypot(F2.real, F2.imag)

plt.clf()
plt.plot(mag0, 'b-', label='Lanczos-3')
plt.plot(mag1, 'g-')
plt.plot(mag2, 'r-', label='"fixed"(2)')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('abs Fourier transform')
plt.xlabel('frequency')
plt.legend()
ps.savefig()

