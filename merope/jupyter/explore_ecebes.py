#!/home/rc8936/.conda/envs/rnc_fes/bin/python

import numpy as np
import h5py
import sys
import re
#import cv2 as cv
#from scipy.fftpack import dct, idct


def getmask(shp):
    mask = [np.zeros(shp, dtype=float) for k in range(4)]
    MASK = [np.zeros(shp, dtype=complex) for k in range(4)]
    mask[0][:3, 1] = 1.
    mask[1][1, :3] = 1.
    mask[2][:3, :3] = np.eye(3)
    mask[3][:3, :3] = np.fliplr(np.eye(3))
    for i in range(len(mask)):
        mask[i] = np.roll(np.roll(mask[i], -1, axis=0), -1, axis=1)
        MASK[i] = np.fft.fft2(mask[i])
    return mask, MASK


def buildfilt(nsamples, nrolls, cutoff=0.01):
    FREQ = np.fft.fftfreq(nsamples)
    filt = np.zeros(FREQ.shape[0], dtype=float)
    inds = np.where(np.abs(FREQ) < cutoff)
    filt[inds] = 0.5 * (1. + np.cos(np.pi * FREQ[inds] / cutoff))
    return np.tile(filt, (nrolls, 1)).T


def getextrema(t1, t2):
    tmin = np.max([np.min(t1), np.min(t2)])
    tmax = np.min([np.max(t1), np.max(t2)])
    return tmin, tmax


def bipolarlognorm(inmat):
    posinds = np.argwhere(inmat > 0)
    neginds = np.argwhere(inmat < 0)
    ypos = np.log(np.abs(inmat[posinds]))
    ypos -= np.min(ypos)
    ypos = (ypos.astype(float) * 254 / np.max(ypos) + 1).astype(int)
    yneg = np.log(np.abs(-1 * inmat[neginds]))
    yneg -= np.min(ypos)
    yneg = (yneg.astype(float) * 254 / np.max(yneg) + 1).astype(int)
    out = np.zeros(inmat.shape, dtype=int)
    out[posinds] = ypos
    out[neginds] = -1 * yneg
    return out


def findroot(th, g):
    # using 2 step Newton-Raphson to find the root that tells me the limit above which the max p = log(histogram value)
    powrange = np.arange(4)
    for i in range(4):
        xpows = np.power(np.full(powrange.shape, g), powrange)
        g -= xpows.dot(th) / (powrange[1:] * xpows[:-1]).dot(th[1:])
    return g


def main():
    # path = '/projects/EKOLEMEN/ecebes'
    # path = '/projects/EKOLEMEN/ecebes'
    path = '/gpfs/slac/staas/fs1/g/coffee_group/edgeml_fes_data/ecebes'
    shot = 180625
    if len(sys.argv) > 1:
        m = re.search('^(.+)(\d{6})$', sys.argv[1])
        if m:
            path = m.group(1)
            shot = int(m.group(2))
    else:
        print('syntax: %s <path/filehead' % sys.argv[0])
    ecefile = '%s/%i%s' % (path, shot, 'ECE')
    besfile = '%s/%i%s' % (path, shot, 'BES')
    outfile = '%s.h5' % (shot)

    with h5py.File(outfile, 'w') as f:
        data_ece = np.load(ecefile, allow_pickle=True)
        data_bes = np.load(besfile, allow_pickle=True)
        chans_ece = list(data_ece.keys())
        chans_bes = list(data_bes.keys())
        t_ece = ((data_ece[chans_ece[0]]['data.time'] + 0.00025) * 1e3).astype(int)
        t_bes = ((data_bes[chans_bes[0]]['data.time'] + 0.00025) * 1e3).astype(int)
        tmin, tmax = getextrema(t_bes, t_ece)
        print(tmin, tmax)
        inds_ece_coince = np.where((t_ece > tmin) * (t_ece < tmax))
        inds_bes_coince = np.where((t_bes > tmin) * (t_bes < tmax))
        sz_ece = t_ece[inds_ece_coince].shape[0]
        sz_bes = t_bes[inds_bes_coince].shape[0]
        print('sz_ece = %i\tsz_bes = %i\tsz_bes-2*sz_ece = %i' % (sz_ece, sz_bes, (sz_bes - 2 * sz_ece)))
        nsamples = 1024
        nfolds = int(sz_ece // nsamples)
        print('nfolds*nsamples = ', nfolds * nsamples)
        t = t_ece[inds_ece_coince[0][:nsamples * nfolds]].reshape(nfolds, nsamples).T
        f.create_dataset('times', data=t)
        grp_ece = f.create_group('ece')

        hbins = [i for i in range(256 + 1)]
        fitlim = int(8)
        hfit = np.array(hbins[1:fitlim])
        hfitmat = np.c_[[int(1) for i in range(fitlim - 1)], hfit, hfit ** 2, hfit ** 3]
        mask, MASK = getmask((nsamples, nfolds))
        for ch in chans_ece:
            m = re.search('^ece.{2}(\d+)$', ch)
            if m:
                print(m.group(1))
                if data_ece[ch]['data.ECE'].shape[0] > 1:
                    x = data_ece[ch]['data.ECE'][inds_ece_coince[0][:nsamples * nfolds]].reshape(nfolds, nsamples).T
                    X = np.fft.fft(x, axis=0)
                    AX = np.abs(X)
                    if np.max(AX) == 0:
                        continue
                    OUT = np.log(np.power(AX, int(2)))
                    BG = np.fft.ifft(np.fft.fft(OUT.copy(), axis=0) * buildfilt(nsamples, nfolds, cutoff=0.05),
                                     axis=0).real
                    OUT = OUT - BG
                    OUT[0, :] = 0
                    OUT -= np.mean(OUT)
                    OUT *= fitlim / np.std(OUT)  # *(OUT>hbins[0])
                    grp_ece.create_dataset(m.group(1), data=OUT[:nsamples // 2, :].astype(np.int16))
                    if False:
                        # disabling the histogramming.

                        '''
                        f(x) = [1,x,x**2,x**3].dot(theta)
                        OK, take the histogram of values (after this mean 0 std 8)... this seems to place modes in example shot at about 80-90 )
                        Take the log of the histogram of values, fit f(x) (zeros centered), to a cubic poly from -1..8 (using peudo inverse method)
                        2 step Newton-Raphson with +8 as initial guess to find root (this is all so far in log space)
                        Find the max of values above the root, call it p
                        scale envelope is defined like Weiner as 1/(1+ exp( f(x) )/exp(p) )
                        '''
                        h = np.log(1 + np.histogram(OUT, hbins)[0])
                        theta = np.linalg.pinv(hfitmat).dot(h[1:fitlim])
                        pows = np.arange(4, dtype=int)
                        fx = np.array(
                            [np.power(np.full(pows.shape, v), pows).dot(theta) for v in OUT.flatten()]).reshape(
                            OUT.shape)  # fitted value for each pixel val
                        '''
                        cutoff = findroot(theta,fitlim+4) # using f(x) to find root using the guess of fitlim as initializer, only two step Newton-Raphson
                        if (cutoff>1) and (cutoff < hbins[-1]):
                            inds = np.where(hbins[:-1]>cutoff)
                            p = np.max(h[inds])
                            print(theta,cutoff,p)
                            OUTFILT = OUT * 1./(1+np.exp(fx) / np.exp(p))
                            #cv.normalize(OUT-BG,OUT,0,255,cv.NORM_MINMAX)
                        '''
                        p = h[fitlim]
                        OUTFILT = OUT * 1. / (1 + np.exp(fx) / np.exp(p))
                        grp_ece.create_dataset('%s_filt' % (m.group(1)),
                                               data=OUTFILT[:nsamples // 2, :].astype(np.uint8))

                    if True:
                        # enabling 3x3 conv kernel
                        # use a 3x3 kernel in pi radian rotation, max val, no pool.
                        QOUT = np.fft.fft2(OUT)
                        NEWOUT = [np.zeros(QOUT.shape, dtype=float) for i in range(len(MASK))]
                        for i in range(len(MASK)):
                            NEWOUT[i] = np.fft.ifft2(QOUT * MASK[i]).real
                        for i in range(len(MASK)):
                            NEWOUT[0] = np.where(NEWOUT[i] > NEWOUT[0], NEWOUT[i], NEWOUT[0])
                        grp_ece.create_dataset('%s_filt' % (m.group(1)),
                                               data=NEWOUT[0][:nsamples // 2, :].astype(np.int16))

                    print(ch, x.shape)
        '''
        print(chans_bes)
        print(chans_bes[-2],(data_bes[chans_bes[-1]]))
        print(chans_bes[-1],(data_bes[chans_bes[-1]]))
        '''

        '''
        grp_bes = f.create_group('bes')
        for ch in chans_bes:
            #if ((type(data_bes[ch]['data.BES'])!=type(None))):# and 
            m = re.search('^bes.{2}(\d+)',ch)
            if m:
                print(m.group(1))
                if (data_bes[ch]['data.BES'].shape[0]>1):
                    x = data_bes[ch]['data.BES'][inds_bes_coince[0][:nsamples*nfolds*2:2]].reshape(nfolds,nsamples).T
                    X = np.fft.fft(x,axis=0)
                    OUT = np.log(np.power(np.abs(X),int(2)))
                    BG = np.fft.ifft(np.fft.fft(OUT.copy(),axis=0) * buildfilt(nsamples,nfolds,cutoff=0.05) ,axis=0).real
                    OUT = OUT - BG
                    OUT -= np.mean(OUT)
                    OUT *= 8/np.std(OUT)
                    #cv.normalize(OUT-BG,OUT,0,255,cv.NORM_MINMAX)
                    grp_bes.create_dataset(m.group(1),data=OUT[:nsamples//2,:].astype(np.int8))
                    print(ch,x.shape)
        '''
        # closing with h5py.File() as f

    return


if __name__ == '__main__':
    main()