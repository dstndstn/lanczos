from __future__ import print_function
import os
import numpy as np
import fitsio
from astrometry.util.util import *
from astrometry.util.file import *
from astrometry.util.plotutils import *
import pylab as plt

'''
Try shifting a source Gaussian through a pixel (one dimension)
and then also resampling by a fraction of a pixel, and measuring
the dcen3x3 centroid.
'''

ps = PlotSequence('dx')

# input image
h,w  = 49,49
xx,yy = np.meshgrid(np.arange(w), np.arange(h))

N1 = N2 = 25

ocen  = np.zeros((N1,N2))
odcen = np.zeros((N1,N2))

ic = np.zeros((N1,N2))

icen = np.zeros(N1)
idcen = np.zeros(N1)

OX = np.linspace(0, 2, N1)
DX = np.linspace(0, 2, N2)

cx,cy = w//2, h//2
print('cx', cx)
for ii,ox in enumerate(OX):

    sig = 3.
    img = np.exp(-0.5 * ((xx - (cx+ox))**2 + (yy - cy)**2) / sig**2)

    pixscale = 0.1
    cd = pixscale / 3600.

    insum = np.sum(img)
    icen[ii] = np.sum(xx * img) / insum

    peak = np.argmax(img)
    py,px = np.unravel_index(peak, img.shape)
    three = img[py-1:py+2, px-1:px+2]
    three = three.astype(float).flat
    result,x,y = dcen3x3b(three[0], three[1], three[2],
                          three[3], three[4], three[5],
                          three[6], three[7], three[8])
    idcen[ii] = x + px - 1

    dy = 0.
    for jj,dx in enumerate(DX):
        hdr = fitsio.FITSHDR()

        ic[ii,jj] = cx + ox + dx
        
        wcs = Tan(0., 0., -dx + 0.5 + h/2., -dy + 0.5 + w/2.,
                  -cd, 0., 0., cd, float(w), float(h))
        wcs.add_to_header(hdr)
        hdr['OBJECT'] = 'INPUT1'
        hdr['GAIN'] = 1.
        fitsio.write('input1.fits', img, header=hdr, clobber=True)

        # output WCS is in swarp.conf -- CRVAL 0,0

        cmd = 'swarp -c swarp.conf input1.fits'
        rtn = os.system(cmd)
        #print('Return:', rtn)
        assert(rtn == 0)
        
        outfn = 'coadd.fits'
        coimg = fitsio.read(outfn)

        outsum = np.sum(coimg)
        ocen[ii,jj] = np.sum(xx * coimg) / outsum

        # Run Blanton's dcen3x3 routine on the un-smoothed image
        peak = np.argmax(coimg)
        py,px = np.unravel_index(peak, coimg.shape)
        three = coimg[py-1:py+2, px-1:px+2]
        three = three.astype(float).flat
        result,x,y = dcen3x3b(three[0], three[1], three[2],
                              three[3], three[4], three[5],
                              three[6], three[7], three[8])
        odcen[ii,jj] = x + px - 1

plt.clf()
plt.plot(OX, icen, 'b-', label='Input centroid')
plt.plot(OX, idcen, 'r-', label='Input dcen3x3')
plt.xlabel('Input subpixel center')
plt.ylabel('Input measured centroid')
plt.legend(loc='upper left')
ps.savefig()

plt.clf()
plt.plot(OX, icen-(OX+cx), 'b-', label='Input centroid')
plt.plot(OX, idcen-(OX+cx), 'r-', label='Input dcen3x3')
plt.xlabel('Input subpixel center')
plt.ylabel('Input measured centroid error')
plt.legend(loc='upper left')
ps.savefig()

plt.clf()
plt.imshow(ocen, interpolation='nearest', origin='lower',
           extent=[DX.min(), DX.max(), OX.min(), OX.max()])
print('ocen', ocen.min(), ocen.max())
plt.xlabel('subpixel shift')
plt.ylabel('input subpixel center')
plt.title('Lanczos-shifted centroid')
plt.colorbar()
ps.savefig()

plt.clf()
plt.imshow(ocen - ic, interpolation='nearest', origin='lower',
           extent=[DX.min(), DX.max(), OX.min(), OX.max()],
           cmap='RdBu')
print('ic', ic.min(), ic.max())
plt.xlabel('subpixel shift')
plt.ylabel('input subpixel center')
plt.title('Lanczos-shifted centroid error')
plt.colorbar()
ps.savefig()

plt.clf()
plt.imshow(odcen, interpolation='nearest', origin='lower',
           extent=[DX.min(), DX.max(), OX.min(), OX.max()])
print('odcen', odcen.min(), odcen.max())
plt.xlabel('subpixel shift')
plt.ylabel('input subpixel center')
plt.title('Lanczos-shifted dcen3x3 result')
plt.colorbar()
ps.savefig()

plt.clf()
plt.imshow(odcen - ic, interpolation='nearest', origin='lower',
           extent=[DX.min(), DX.max(), OX.min(), OX.max()],
           cmap='RdBu')
plt.xlabel('subpixel shift')
plt.ylabel('input subpixel center')
plt.title('Lanczos-shifted dcen3x3 error')
plt.colorbar()
ps.savefig()


plt.clf()
plt.imshow(odcen - ocen, interpolation='nearest', origin='lower',
           extent=[DX.min(), DX.max(), OX.min(), OX.max()],
           cmap='RdBu')
plt.xlabel('subpixel shift')
plt.ylabel('input subpixel center')
plt.title('Lanczos-shifted dcen3x3 - Lanczos-shifted centroid')
plt.colorbar()
ps.savefig()
