from __future__ import print_function
import os
import numpy as np
import fitsio
from astrometry.util.util import *
from astrometry.util.fits import fits_table
from astrometry.util.file import *
from astrometry.util.plotutils import *
import pylab as plt
from tractor import *


ps = PlotSequence('dx')

# input image
h,w  = 100,100
xx,yy = np.meshgrid(np.arange(w), np.arange(h))
sig = 3.

N = 10

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

    # run psfex on the SE catalog

    # PSF_RECENTER?
    # CENTER_KEYS?

    # se2.conf : PHOT_APERTURES

    cmd = 'psfex -c psfex.conf se.fits -PSF_SUFFIX .psfex'
    rtn = os.system(cmd)

    print('Return:', rtn)
    assert(rtn == 0)

