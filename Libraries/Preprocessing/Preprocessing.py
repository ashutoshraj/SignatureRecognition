from Libraries.StrokeFilter.StrokeFilter import strokeFilter
from Libraries.StrokeFrequency.StrokeFrequency import strokeFreq
from Libraries.StrokeOrient.StrokeOrient import strokeOrient
from Libraries.StrokeSegmentation.StrokeSegmentation import strokeSeg


def preprocessing(img):
    blksze = 16
    thresh = 0.3
    normim, mask = strokeSeg(img, blksze, thresh)

    gradientsigma = 1
    blocksigma = 7
    orientsmoothsigma = 7
    orientim = strokeOrient(normim, gradientsigma, blocksigma, orientsmoothsigma)

    blksze = 38
    windsze = 5
    minWaveLength = 5
    maxWaveLength = 15
    freq, medfreq = strokeFreq(normim, mask, orientim, blksze, windsze, minWaveLength, maxWaveLength)

    freq = medfreq * mask
    kx = 0.65
    ky = 0.65
    newim = strokeFilter(normim, orientim, freq, kx, ky)
    return newim < -3
