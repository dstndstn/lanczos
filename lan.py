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

#xx = np.arange(-4., 3+1.).astype(np.float32)
xx = np.arange(-6., 5+1.).astype(np.float32)
#xx += 0.5 * (1./subsampling)

lanczos_kernel = np.zeros((len(xx) * subsampling))
lanczos5_kernel = np.zeros((len(xx) * subsampling))
fixed_kernel = np.zeros((len(xx) * subsampling))
fixed2_kernel = np.zeros((len(xx) * subsampling))
fixedi_kernel = np.zeros((len(xx) * subsampling))

subx = np.zeros((len(xx) * subsampling))

fixed_x = np.zeros(subsampling)
fixed_x2 = np.zeros(subsampling)
fixed_xi = np.zeros(subsampling)

cenu = np.zeros(subsampling)
cen0 = np.zeros(subsampling)
cen1 = np.zeros(subsampling)
cen2 = np.zeros(subsampling)
ceni = np.zeros(subsampling)

cenp = np.zeros(subsampling)
fixedp_kernel = np.zeros((len(xx) * subsampling))

DX = np.linspace(0, 1, subsampling, endpoint=False)
for i,dx in enumerate(DX):
    L = np.zeros_like(xx)
    rtn = lanczos3_filter(xx + dx, L)
    assert(rtn == 0)
    lanczos_kernel[i::subsampling] = L

    cenu[i] = -np.sum(L * xx)

    L /= L.sum()

    L5 = np.zeros_like(xx)
    rtn = lanczos5_filter(xx + dx, L5)
    assert(rtn == 0)
    lanczos5_kernel[i::subsampling] = L5
    L5 /= L5.sum()
    
    subx[i::subsampling] = xx + dx
    
    print('dx', dx)
    #Lsum = L.sum()
    #print('sum', L.sum())
    centroid = -np.sum(L * xx)# / Lsum
    print('centroid', centroid, 'vs', dx)
    cen0[i] = centroid

    fx = dx - (centroid - dx)
    rtn = lanczos3_filter(xx + fx, L)
    L /= L.sum()
    assert(rtn == 0)
    fixed_kernel[i::subsampling] = L
    #print('sum', L.sum())
    centroid = -np.sum(L * xx) #/ Lsum
    print('centroid', centroid, 'vs', dx)
    cen1[i] = centroid
    fixed_x[i] = fx
    
    fx2 = fx - (centroid - dx)
    rtn = lanczos3_filter(xx + fx2, L)
    L /= L.sum()
    assert(rtn == 0)
    fixed2_kernel[i::subsampling] = L
    fixed_x2[i] = fx2
    #print('sum', L.sum())
    centroid = -np.sum(L * xx)# / Lsum
    print('centroid', centroid, 'vs', dx)
    cen2[i] = centroid

    fxi = dx
    for ii in range(8):
        rtn = lanczos3_filter(xx + fxi, L)
        assert(rtn == 0)
        L /= L.sum()
        centroid = -np.sum(L * xx)
        fxinext = fxi - (centroid - dx)
        print('  dx %.8f -> centroid %.8f (err %8.2g) via fxi %.8f, fxi_next %.8f' %
              (dx, centroid, centroid-dx, fxi, fxinext))

        fixedi_kernel[i::subsampling] = L
        fixed_xi[i] = fxi
        ceni[i] = centroid

        fxi = fxinext

# Fit a polynomial expansion to fixed_xi
NP = 10
A = np.zeros((len(DX), NP+1))
for i in range(NP+1):
    A[:,i] = DX**i
R = np.linalg.lstsq(A, fixed_xi - DX)
poly = R[0]
print('Fitting polynomial:', poly)


# # Read off a piecewise linear relation
# nn = 8
# dxsup = np.hstack((DX, [1.0]))
# fxsup = np.hstack((fixed_xi - DX, [fixed_xi[0]-DX[0]]))
# print('aug DX', dxsup)
# print('aug fx', fxsup)



for i,dx in enumerate(DX):
    fx = np.sum([a * dx**j for j,a in enumerate(poly)])
    rtn = lanczos3_filter(xx + dx + fx, L)
    assert(rtn == 0)
    L /= L.sum()
    centroid = -np.sum(L * xx)
    cenp[i] = centroid
    fixedp_kernel[i::subsampling] = L

print('subx', subx.min(), subx.max())

plt.clf()
plt.plot(subx, lanczos_kernel, 'b-', lw=3, alpha=0.5, label='Lanczos-3')
#plt.plot(fixed_kernel, 'r-')
plt.plot(subx, fixed2_kernel, 'r-', label='"fixed(2)"')
plt.plot(subx, fixedi_kernel, 'm-', label='"fixed(5)"')
plt.plot(subx, fixedp_kernel, '-', color='0.5', label='"fixed(poly)"')
plt.plot(subx, lanczos5_kernel, 'g-', alpha=0.5, label='Lanczos-5')
#plt.xlim(-3.5, 3.5)
plt.xlim(-5.5, 5.5)
plt.legend()
ps.savefig()
    
plt.clf()
plt.plot(DX, cen0, 'b-')
plt.plot(DX, cen1, 'g-')
plt.plot(DX, cen2, 'r-')
plt.plot(DX, ceni, 'm-')
plt.plot(DX, cenp, '-', color='0.5')
ps.savefig()

plt.clf()
plt.plot(DX, cenu - DX, 'k-', label='Lanczos-3 (unnorm)')
plt.plot(DX, cen0 - DX, 'b-', label='Lanczos-3')
plt.plot(DX, cen1 - DX, 'g-', label='"fixed(1)"')
plt.plot(DX, cen2 - DX, 'r-', label='"fixed(2)"')
plt.plot(DX, ceni - DX, 'm-', label='"fixed(5)"')
plt.plot(DX, cenp - DX, '-', color='0.5', label='"fixed(poly)"')
plt.xlabel('subpixel offset (pixels)')
plt.ylabel('centroid error (pixels)')
plt.legend()
ps.savefig()

plt.clf()
plt.plot(DX, cenp - DX, '-', color='0.5', label='"fixed(poly)"')
plt.xlabel('subpixel offset (pixels)')
plt.ylabel('centroid error (pixels)')
plt.legend()
ps.savefig()

# Fit a polynomial expansion to fixed_xi
# NP = 10
# A = np.zeros((len(DX), NP+1))
# for i in range(NP+1):
#     A[:,i] = DX**i
# R = np.linalg.lstsq(A, fixed_xi - DX)
# poly = R[0]
# print('Fitting polynomial:', poly)

fipoly = np.zeros_like(DX)
for i,a in enumerate(poly):
    fipoly += a * DX**i
    
plt.clf()
#plt.plot(DX, fixed_x - DX, 'g-')
#plt.plot(DX, fixed_x2 - DX, 'r-', lw=3)
plt.plot(DX, fixed_xi - DX, 'm-', label='x_i')
plt.plot(DX, fipoly, 'b-', label='polynomial')
plt.xlabel('subpixel offset (pixels)')
plt.ylabel('"fixed" Lanczos argument correction (pixels)')
plt.legend(loc='upper right')
ps.savefig()

F0 = np.fft.rfft(lanczos_kernel)
F5 = np.fft.rfft(lanczos5_kernel)
F1 = np.fft.rfft(fixed_kernel)
F2 = np.fft.rfft(fixed2_kernel)
Fi = np.fft.rfft(fixedi_kernel)
Fp = np.fft.rfft(fixedp_kernel)

print(len(lanczos_kernel))

plt.clf()
plt.subplot(2,1,1)
plt.plot(F0.real, 'b-')
plt.plot(F1.real, 'g-')
plt.plot(F2.real, 'r-')
plt.plot(F5.real, 'm-')
plt.plot(Fi.real, 'c-')
plt.plot(Fp.real, '-', color='0.5')
plt.xscale('log')
plt.ylabel('Fourier transform (real)')
plt.subplot(2,1,2)
plt.plot(F0.imag, 'b-')
plt.plot(F1.imag, 'g-')
plt.plot(F2.imag, 'r-')
plt.plot(F5.imag, 'm-')
plt.plot(Fi.imag, 'c-')
plt.plot(Fp.imag, '-', color='0.5')
plt.xscale('log')
plt.ylabel('Fourier transform (imag)')
ps.savefig()

mag0 = np.hypot(F0.real, F0.imag)
mag5 = np.hypot(F5.real, F5.imag)
mag1 = np.hypot(F1.real, F1.imag)
mag2 = np.hypot(F2.real, F2.imag)
magi = np.hypot(Fi.real, Fi.imag)
magp = np.hypot(Fp.real, Fp.imag)

plt.clf()
plt.plot(mag0, 'b-', label='Lanczos-3')
plt.plot(mag1, 'g-')
plt.plot(mag2, 'r-', label='"fixed"(2)')
plt.plot(magi, 'c-', label='"fixed"(5)')
plt.plot(mag5, 'm-', label='Lanczos-5')
plt.plot(magp, '-', color='0.5', label='Fixed-poly')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('abs Fourier transform')
plt.xlabel('frequency')
plt.legend()
ps.savefig()

