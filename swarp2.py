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

'''
Try shifting a source Gaussian through a pixel (in one dimension) and
then also resampling by a fraction of a pixel (in the same dimension),
and measuring the dcen3x3 centroid.
'''

ps = PlotSequence('sw2')

# input image
h,w  = 49,49
#h,w  = 25,25
#h,w  = 15,15
xx,yy = np.meshgrid(np.arange(w), np.arange(h))

#N1 = N2 = 25
#N1 = 25
#N2 = 23
N1 = 5
N2 = 11

#alanczos = True
alanczos = False

ocen  = np.zeros((N1,N2))
odcen = np.zeros((N1,N2))
osecen = np.zeros((N1,N2))
osewcen = np.zeros((N1,N2))
otcen = np.zeros((N1,N2))

ic = np.zeros((N1,N2))

icen = np.zeros(N1)
idcen = np.zeros(N1)
isecen = np.zeros(N1)
isewcen = np.zeros(N1)
itcen = np.zeros(N1)

#OX = np.linspace(0, 2, N1)
#DX = np.linspace(0, 2, N2)

OX = np.linspace(0, 1, N1)
DX = np.linspace(0, 1, N2)

cx,cy = w//2, h//2
print('cx', cx)
for ii,ox in enumerate(OX):
    print('Gaussian centered at', ox, ', ', ii+1, 'of', len(OX))
    sig = 3.
    #sig = 4.
    #sig = 2.5
    img = np.exp(-0.5 * ((xx - (cx+ox))**2 + (yy - cy)**2) / sig**2)

    #img += np.random.normal(size=img.shape)*0.001
    
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

    fitsio.write('input1.fits', img, clobber=True)

    # plt.clf()
    # plt.imshow(img, interpolation='nearest', origin='lower')
    # plt.title('Input: ox=%g' % ox)
    # ps.savefig()
    
    # run source extractor on the input image
    cmd = 'sex -c se.conf input1.fits'
    rtn = os.system(cmd)
    #print('Return:', rtn)
    # -> se.fits
    assert(rtn == 0)
    T = fits_table('se.fits')
    #print(len(T), 'sources')
    assert(len(T) == 1)
    isecen[ii] = T.x_image[0] - 1.
    isewcen[ii] = T.xwin_image[0] - 1.

    # run the tractor on the input image
    tim = Image(data=img, inverr=np.ones_like(img),
                psf=NCircularGaussianPSF([sig],[1.]))
    pos = PixPos(cx,cy)
    pos.stepsizes = [1e-3, 1e-3]
    pos.symmetric_derivs = True
    src = PointSource(pos, Flux(1.))
    tractor = Tractor([tim], [src])
    tractor.freezeParam('images')
    tractor.optimize_loop()
    itcen[ii] = src.pos.x
    
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

        if alanczos:
            resamp = 'ALANCZOS3'
        else:
            resamp = 'LANCZOS3'
        cmd = 'swarp -c swarp.conf -RESAMPLING_TYPE %s -IMAGE_SIZE %i,%i input1.fits' % (resamp, w, h)
        rtn = os.system(cmd)
        #print('Return:', rtn)
        assert(rtn == 0)
        
        outfn = 'coadd.fits'
        coimg = fitsio.read(outfn)

        # plt.clf()
        # plt.imshow(coimg, interpolation='nearest', origin='lower')
        # plt.title('Resampled: ox=%g, dx=%g' % (ox,dx))
        # ps.savefig()
        
        shiftedimg = np.exp(-0.5 * ((xx - (cx+ox+dx))**2 + (yy - cy)**2) / sig**2)
        # Visually they look the same
        # plt.clf()
        # plt.imshow(coimg/coimg.sum(), interpolation='nearest', origin='lower')
        # plt.colorbar()
        # plt.title('coimg: dx %f' % dx)
        # ps.savefig()
        # 
        # plt.clf()
        # plt.imshow(shiftedimg/shiftedimg.sum(), interpolation='nearest', origin='lower')
        # plt.colorbar()
        # plt.title('shifted: dx %f' % dx)
        # ps.savefig()
        
        # plt.clf()
        # plt.imshow(coimg/coimg.sum() - shiftedimg/shiftedimg.sum(), interpolation='nearest', origin='lower')
        # plt.colorbar()

        # print('Centroid of image differences:',
        #       np.sum(xx * (coimg/coimg.sum() - shiftedimg/shiftedimg.sum())))
        
        # plt.title('co-shifted: dx %f (Gaussian max: %f)' % (dx, (coimg/coimg.sum()).max()))
        # ps.savefig()
        
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

        # run source extractor on the output (resampled) image
        cmd = 'sex -c se.conf %s' % outfn
        #print(cmd)
        rtn = os.system(cmd)
        assert(rtn == 0)
        T = fits_table('se.fits')
        #print(len(T), 'sources in resampled')
        assert(len(T) == 1)
        osecen [ii,jj] = T.x_image   [0] - 1.
        osewcen[ii,jj] = T.xwin_image[0] - 1.

        # run the tractor on the output (resampled) image.  We re-use
        # the data structures from above, AND the position from the
        # previous fit, but that's okay because the 'coimg' pixels are
        # different each time, and the initialization doesn't matter
        # much.
        tim.data = coimg
        tractor.optimize_loop()
        otcen[ii,jj] = src.pos.x
        

# plt.clf()
# plt.plot(OX, icen, 'b-', label='Input centroid')
# plt.plot(OX, idcen, 'r-', label='Input dcen3x3')
# plt.plot(OX, isecen, 'g-', label='Input SE x_image')
# plt.plot(OX, isewcen, 'm-', label='Input SE xwin_image')
# plt.plot(OX, itcen, 'c-', label='Input tractor pos')
# plt.xlabel('Input subpixel center')
# plt.ylabel('Input measured centroid')
# plt.legend(loc='upper left')
# ps.savefig()

plt.clf()
plt.plot(OX, icen-(OX+cx), 'b-', label='Input centroid (max %.1g)' %
         (np.max(np.abs(icen-(OX+cx)))))
plt.plot(OX, idcen-(OX+cx), 'r-', label='Input dcen3x3 (max %.1g)' %
         (np.max(np.abs(idcen-(OX+cx)))))
plt.plot(OX, isecen-(OX+cx), 'g-', label='Input SE x_image (max %.1g)' %
         (np.max(np.abs(isecen-(OX+cx)))))
plt.plot(OX, isewcen-(OX+cx), 'm-', label='Input SE xwin_image (max %.1g)' %
         (np.max(np.abs(isewcen-(OX+cx)))))
plt.plot(OX, itcen-(OX+cx), 'm-', label='Input tractor pos (max %.1f)' %
         (np.max(np.abs(itcen-(OX+cx)))))
plt.xlabel('Input subpixel center')
plt.ylabel('Input measured centroid error')
plt.legend(loc='upper left')
ps.savefig()


# plt.clf()
# plt.plot(OX, icen-(OX+cx), 'b-', label='Input centroid')
# #plt.plot(OX, idcen-(OX+cx), 'r-', label='Input dcen3x3')
# plt.plot(OX, isecen-(OX+cx), 'g-', label='Input SE x_image')
# plt.plot(OX, isewcen-(OX+cx), 'm-', label='Input SE xwin_image')
# plt.plot(OX, itcen-(OX+cx), 'm-', label='Input tractor pos')
# plt.xlabel('Input subpixel center')
# plt.ylabel('Input measured centroid error')
# plt.legend(loc='upper left')
# ps.savefig()

#plt.yscale('symlog')
#ps.savefig()

plt.clf()
plt.imshow(ocen, interpolation='nearest', origin='lower',
           extent=[DX.min(), DX.max(), OX.min(), OX.max()])
print('ocen', ocen.min(), ocen.max())
plt.xlabel('subpixel shift')
plt.ylabel('input subpixel center')
plt.title('Lanczos-shifted centroid')
plt.colorbar()
ps.savefig()

for name,img in [('centroid', ocen), ('dcen3x3', odcen),
                 ('SE x_image', osecen), ('SE xwin_image', osewcen),
                 ('Tractor (w/ correct PSF)', otcen)]:
    plt.clf()
    absmax = np.max(np.abs(img - ic))
    plt.imshow(img - ic, interpolation='nearest', origin='lower',
               extent=[DX.min(), DX.max(), OX.min(), OX.max()],
               cmap='RdBu', vmin=-absmax, vmax=absmax)
    plt.xlabel('subpixel shift')
    plt.ylabel('input subpixel center')
    plt.title('Error in Lanczos-shifted center: %s' % name)
    plt.colorbar()
    ps.savefig()


plt.clf()
for name,img in [('centroid', ocen),
                 ('SE x_image', osecen), ('SE xwin_image', osewcen),
                 ('Tractor (w/ correct PSF)', otcen)]:
    mean = np.mean(img - ic, axis=0)
    std  = np.std (img - ic, axis=0)
    plt.errorbar(DX, mean, yerr=std, fmt='.-', label=name)
plt.xlabel('Subpixel shift (pixels)')
plt.ylabel('Centroid error (pixels)')
if alanczos:
    plt.title('Astrometric-Lanczos-3 shift')
else:
    plt.title('Vanilla Lanczos-3 shift')
plt.legend(loc='upper right')
ps.savefig()
    
# plt.clf()
# plt.imshow(odcen - ocen, interpolation='nearest', origin='lower',
#            extent=[DX.min(), DX.max(), OX.min(), OX.max()],
#            cmap='RdBu')
# plt.xlabel('subpixel shift')
# plt.ylabel('input subpixel center')
# plt.title('Lanczos-shifted dcen3x3 - Lanczos-shifted centroid')
# plt.colorbar()
# ps.savefig()
