from __future__ import print_function
import os
import numpy as np
import fitsio
from astrometry.util.util import *
from astrometry.util.file import *
import pylab as plt

# input image
#h,w  = 99,99
h,w  = 49,49
xx,yy = np.meshgrid(np.arange(w), np.arange(h))
#cx,cy = w//2, h//2
cx,cy = 24.5, 24.25
sig = 3.
img = np.exp(-0.5 * ((xx - cx)**2 + (yy - cy)**2) / sig**2)

pixscale = 0.1
#dx,dy = 0.1, 0.
cd = pixscale / 3600.

outcenx = []
incenx = []
outceny = []
inceny = []

dcenx = []
dceny = []

#DX = np.linspace(-1, 1, 256)
#DX = np.linspace(-1, 1, 100)
DX = np.linspace(-1, 1, 25)
DY = np.linspace(-1, 1, 25)

for dx,dy in zip(DX,DY):
    hdr = fitsio.FITSHDR()

    # s = -0.0201157 * np.sin(2.*np.pi * dx)
    # s2 = -0.001409137 * np.sin(2. * 2. * np.pi * dx)
    # dx -= s + s2
    
    wcs = Tan(0., 0., -dx + 0.5 + h/2., -dy + 0.5 + w/2.,
              -cd, 0., 0., cd, float(w), float(h))
    wcs.add_to_header(hdr)
    hdr['OBJECT'] = 'INPUT1'
    hdr['GAIN'] = 1.
    fitsio.write('input1.fits', img, header=hdr, clobber=True)

    # output WCS is in swarp.conf -- CRVAL 0,0

    cmd = 'swarp -c swarp.conf input1.fits'
    print('Running:', cmd)
    rtn = os.system(cmd)
    print('Return:', rtn)

    outfn = 'coadd.fits'
    coimg = fitsio.read(outfn)

    insum = np.sum(img)
    print('Input centroid:',
          np.sum(xx * img) / insum, np.sum(yy * img) / insum)
    print('Input sum:', insum)
    incenx.append(np.sum(xx * img) / insum)
    inceny.append(np.sum(yy * img) / insum)

    outsum = np.sum(coimg)
    print('Output centroid:',
          np.sum(xx * coimg) / outsum, np.sum(yy * coimg) / outsum)
    print('Output sum', outsum)
          
    outcenx.append(np.sum(xx * coimg) / outsum)
    outceny.append(np.sum(yy * coimg) / outsum)

    # Run Blanton's dcen3x3 routine on the un-smoothed image
    peak = np.argmax(coimg)
    py,px = np.unravel_index(peak, coimg.shape)
    three = coimg[py-1:py+2, px-1:px+2]
    #print('three:', three.shape, three.dtype)
    three = three.astype(float).flat
    result,x,y = dcen3x3b(three[0], three[1], three[2],
                          three[3], three[4], three[5],
                          three[6], three[7], three[8])
    #print('Result', result, 'x,y', x,y)
    dcenx.append(x + px - 1)
    dceny.append(y + py - 1)

incenx = np.array(incenx)
inceny = np.array(inceny)
outcenx = np.array(outcenx)
outceny = np.array(outceny)
dcenx = np.array(dcenx)
dceny = np.array(dceny)
    
plt.clf()
#plt.plot(DX, incenx, 'b.')
#plt.plot(DX, outcenx, 'r.')
plt.plot(DX, outcenx, 'b-')
plt.plot(DY, outceny, 'g-')
plt.xlabel('dx (input)')
plt.ylabel('x centroid (output)')
plt.savefig('dx1.png')

plt.clf()
plt.plot(DX, outcenx - DX - cx, 'b-')
plt.plot(DY, outceny - DY - cy, 'g-')
plt.title('Center of light errors')
plt.savefig('dx2.png')

plt.clf()
plt.plot(DX, dcenx, 'b-')
plt.plot(DY, dceny, 'g-')
plt.savefig('dx3.png')

plt.clf()
plt.plot(DX, dcenx - DX - cx, 'b-')
plt.plot(DY, dceny - DY - cy, 'g-')
plt.title('dcen3x3 centroid errors')
plt.savefig('dx4.png')


# # ~ 0.02 peak-to-peak
# s = -0.0201157 * np.sin(2.*np.pi * DX)
# plt.clf()
# plt.plot(DX, (outcenx-cx) - DX, 'b-')
# plt.plot(DX, s, 'r-')
# plt.savefig('dx3.png')
# 
# # ~ 0.002 peak-to-peak
# s2 = -0.001409137 * np.sin(2. * 2. * np.pi * DX)
# plt.clf()
# plt.plot(DX, (outcenx-cx) - DX - s, 'b-')
# plt.plot(DX, s2, 'r-')
# plt.ylabel('cenx - sine2 resid')
# plt.savefig('dx4.png')
# 
# # ~ 0.0008 ptp
# s3 = -0.0003478733 * np.sin(3. * 2. * np.pi * DX)
# plt.clf()
# plt.plot(DX, (outcenx-cx) - DX - s - s2, 'b-')
# plt.plot(DX, s3, 'r-')
# plt.ylabel('cenx - sine resid')
# plt.savefig('dx5.png')
# 
# # ~ 0.0004 p2p
# plt.clf()
# plt.plot(DX, (outcenx-cx) - DX - s - s2 - s3, 'b-')
# plt.ylabel('cenx - sine resid')
# plt.savefig('dx6.png')

