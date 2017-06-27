from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import fitsio
from astrometry.util.util import *
from astrometry.util.fits import fits_table
from astrometry.util.file import *
from astrometry.util.plotutils import *
from astrometry.util.run_command import run_command
    
import pylab as plt
from tractor import *

import photutils

ps = PlotSequence('dx')

# input image
#h,w  = 200,200
#h,w  = 500,500
h,w  = 1000,1000
sig = 2.
# per-pix noise
sig1 = 0.000001

N = 20

container = '2b6b936ffec5981b9d0aaea5073878578e651597e7a3374152f70c5ac368bb29'

# cmd = 'docker create -i dstn/astro'
# print(cmd)
# rtn,out,err = run_command(cmd)
# assert(rtn == 0)
# container = out.strip()
# print('rtn', rtn)
# print('out', out)
# print('err', err)
# 
# cmd = 'docker start %s' % container
# print(cmd)
# rtn = os.system(cmd)
# print('rtn', rtn)

# integers
dy = h // N
dx = w // N
Y = dy//2 + dy * np.arange(N)
X = dx//2 + dx * np.arange(N)

#xx,yy = np.meshgrid(np.arange(-dx//2, dx//2+1), np.arange(-dy//2, dy//2+1))
xx,yy = np.meshgrid(np.arange(-dx//2, dx//2), np.arange(-dy//2, dy//2))

flags = np.zeros((h,w), np.int16)
fitsio.write('flag.fits', flags, clobber=True)

cogs = []

for randomize in [False, True]:
    img = np.zeros((h,w), np.float32)
    for y in Y:
        for x in X:
            cx,cy = x,y
            if randomize:
                cy += np.random.uniform(-0.5, +0.5)
                cx += np.random.uniform(-0.5, +0.5)
            sh,sw = yy.shape
            y0,x0 = y-dy//2, x-dx//2
            img[y0:y0+sh, x0:x0+sw] += np.exp(-0.5 * ((cx - (xx+x))**2 + (cy - (yy+y))**2) / sig**2)
            
    img += np.random.normal(scale=sig1, size=img.shape)
    img += 1.
    
    plt.clf()
    plt.imshow(img, interpolation='nearest', origin='lower')
    ps.savefig()

    fitsio.write('input1.fits', img, clobber=True)

    # run source extractor on the input image
    cmd = 'sex -c se2.conf input1.fits'
    
    rtn = os.system(cmd)
    #print('Return:', rtn)
    # -> se.fits
    assert(rtn == 0)

    # FITS_LDAC
    #T = fits_table('se.fits')
    T = fits_table('se.fits', ext=2)
    N = T.vignet.shape[0]
    #print('Vignette range', T.vignet.max())
    plt.clf()
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(T.vignet[i,:,:], interpolation='nearest', origin='lower', vmin=-0.1, vmax=1.0)
    plt.suptitle('Vignettes')
    ps.savefig()
        
    # run psfex on the SE catalog

    # PSF_RECENTER?
    # CENTER_KEYS?

    # se2.conf : PHOT_APERTURES

    for fn in ['psfex.conf', 'se.fits']:
        cmd = 'docker cp %s %s:/' % (fn, container)
        print(cmd)
        rtn = os.system(cmd)
        print('rtn', rtn)

    cmd = 'psfex -c psfex.conf se.fits'
    cmd = 'docker exec %s %s' % (container, cmd)
    print(cmd)
    rtn = os.system(cmd)
    print('rtn', rtn)
    
    #cmd = 'psfex -c psfex.conf se.fits -PSF_SUFFIX .psfex'
    #rtn = os.system(cmd)

    print('Return:', rtn)
    assert(rtn == 0)

    for fn in ['se.psfex', 'snap_imres_se.fits', 'chi_se.fits', 'snap_se.fits',
               'samp_se.fits', 'resi_se.fits', 'proto_se.fits',
        #'residuals_se.ps', 'chi2_se.ps'
        ]:
        cmd = 'docker cp %s:/%s .' % (container, fn)
        print(cmd)
        rtn = os.system(cmd)
        print('rtn', rtn)
    
    T = fits_table('se.psfex')
    psf = T.psf_mask[0]
    print('PSF model:', psf.shape)

    N,ph,pw = psf.shape

    for i in range(N):
        plt.clf()
        if i == 0:
            plt.imshow(psf[i,:,:], interpolation='nearest', origin='lower')
        else:
            mx = np.max(np.abs(psf[i,:,:]))
            plt.imshow(psf[i,:,:], interpolation='nearest', origin='lower',
                       vmin=-mx, vmax=mx)
        plt.title('PSF component %i' % i)
        plt.colorbar()
        ps.savefig()

    for fn in ['resi_se.fits', 'chi_se.fits']:
        imxx = fitsio.read(fn)
        plt.clf()
        plt.imshow(imxx, interpolation='nearest', origin='lower')
        plt.colorbar()
        plt.title('%s' % fn)
        ps.savefig()
        
    cog = []
    cog_radius = np.arange(0.5, 15.5, 0.5)
    for r in cog_radius:
        aper = photutils.CircularAperture(np.array([[pw//2, ph//2]]), r)
        p = photutils.aperture_photometry(psf[0,:,:], aper)
        #print('Ap-phot result:', p)
        #ap = p.field('aperture_sum')
        #print('ap', ap)
        cog.append(p.field('aperture_sum')[0])
    cogs.append(cog)

    plt.clf()
    plt.plot(cog_radius, cog, 'b-')
    plt.xlabel('radius')
    plt.ylabel('PSF component 0 aperture flux')
    ps.savefig()

plt.clf()
plt.plot(cog_radius, cogs[0], 'b-', label='Centered PSFs')
plt.plot(cog_radius, cogs[1], 'r-', label='Randomly scattered PSFs')
plt.xlabel('radius')
plt.ylabel('PSF component 0 aperture flux')
ps.savefig()

    
