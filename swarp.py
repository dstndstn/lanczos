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
cx,cy = 24.5, 24.
sig = 3.
img = np.exp(-0.5 * ((xx - cx)**2 + (yy - cy)**2) / sig**2)

pixscale = 0.1
dx,dy = 0.1, 0.
cd = pixscale / 3600.

outcenx = []
incenx = []
#DX = np.linspace(-1, 1, 256)
DX = np.linspace(-1, 1, 100)

for dx in DX:
    hdr = fitsio.FITSHDR()
    wcs = Tan(0., 0., -dx + 0.5 + h/2., -dy + 0.5 + w/2.,
              -cd, 0., 0., cd, w, h)
    wcs.add_to_header(hdr)
    hdr['OBJECT'] = 'INPUT1'
    hdr['GAIN'] = 1.
    fitsio.write('input1.fits', img, header=hdr, clobber=True)

    # output WCS is in swarp.conf -- CRVAL 0,0

    cmd = 'swarp -c swarp.conf input1.fits'
    rtn = os.system(cmd)
    print('Return:', rtn)

    outfn = 'coadd.fits'
    coimg = fitsio.read(outfn)

    insum = np.sum(img)
    print('Input centroid:',
          np.sum(xx * img) / insum, np.sum(yy * img) / insum)
    print('Input sum:', insum)
    incenx.append(np.sum(xx * img) / insum)

    outsum = np.sum(coimg)
    print('Output centroid:',
          np.sum(xx * coimg) / outsum, np.sum(yy * coimg) / outsum)
    print('Output sum', outsum)
          
    outcenx.append(np.sum(xx * coimg) / outsum)

incenx = np.array(incenx)
outcenx = np.array(outcenx)
    
plt.clf()
#plt.plot(DX, incenx, 'b.')
#plt.plot(DX, outcenx, 'r.')
plt.plot(DX, outcenx, 'b-')
plt.xlabel('dx (input)')
plt.ylabel('x centroid (output)')
plt.savefig('dx1.png')

plt.clf()
plt.plot(DX, outcenx - DX, 'b-')
plt.savefig('dx2.png')

# ~ 0.02 peak-to-peak
s = -0.0201157 * np.sin(2.*np.pi * DX)
plt.clf()
plt.plot(DX, (outcenx-cx) - DX, 'b-')
plt.plot(DX, s, 'r-')
plt.savefig('dx3.png')

# ~ 0.002 peak-to-peak
s2 = -0.001409137 * np.sin(2. * 2. * np.pi * DX)
plt.clf()
plt.plot(DX, (outcenx-cx) - DX - s, 'b-')
plt.plot(DX, s2, 'r-')
plt.ylabel('cenx - sine2 resid')
plt.savefig('dx4.png')

# ~ 0.0008 ptp
s3 = -0.0003478733 * np.sin(3. * 2. * np.pi * DX)
plt.clf()
plt.plot(DX, (outcenx-cx) - DX - s - s2, 'b-')
plt.plot(DX, s3, 'r-')
plt.ylabel('cenx - sine resid')
plt.savefig('dx5.png')

# ~ 0.0004 p2p
plt.clf()
plt.plot(DX, (outcenx-cx) - DX - s - s2 - s3, 'b-')
plt.ylabel('cenx - sine resid')
plt.savefig('dx6.png')

