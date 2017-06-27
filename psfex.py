from __future__ import print_function
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


ps = PlotSequence('dx')

# input image
h,w  = 200,200
xx,yy = np.meshgrid(np.arange(w), np.arange(h))
sig = 2.

N = 10

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


for randomize in [False, True]:
    img = np.zeros((h,w), np.float32)
    for y in np.linspace(0, h, N):
        y += 0.5*h/N
        for x in np.linspace(0, w, N):
            x += 0.5*w/N
        
            if randomize:
                y += np.rand.uniform(-0.5, +0.5)
                x += np.rand.uniform(-0.5, +0.5)

            img += np.exp(-0.5 * ((xx - x)**2 + (yy - y)**2) / sig**2)

    img += np.random.normal(scale=0.01, size=img.shape)
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

    T = fits_table('se.fits')
    N = T.vignet.shape[0]
    print('Vignette range', T.vignet.max())
    for i in range(N):
        plt.clf()
        plt.imshow(T.vignet[i,:,:], interpolation='nearest', origin='lower', vmin=-0.1, vmax=1.0)
        plt.colorbar()
        #plt.savefig('/tmp/v%02i.png' % i)
        ps.savefig()
        print('Wrote vignette', i)
        break
        
    # run psfex on the SE catalog

    # PSF_RECENTER?
    # CENTER_KEYS?

    # se2.conf : PHOT_APERTURES

    for fn in ['psfex.conf', 'psfex-default.conf', 'se.fits']:
        cmd = 'docker cp %s %s:/' % (fn, container)
        print(cmd)
        rtn = os.system(cmd)
        print('rtn', rtn)

    cmd = 'psfex -c psfex.conf se.fits -PSF_SUFFIX .psfex'
    #cmd = 'psfex -c psfex-default.conf se.fits -PSF_SUFFIX .psfex'
    cmd = 'docker exec %s %s' % (container, cmd)
    print(cmd)
    rtn = os.system(cmd)
    print('rtn', rtn)
    
    #cmd = 'psfex -c psfex.conf se.fits -PSF_SUFFIX .psfex'
    #rtn = os.system(cmd)

    print('Return:', rtn)
    assert(rtn == 0)

