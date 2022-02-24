import numpy
from featureExtraction import sigproc
from scipy.fftpack import dct,fft, ifft
import math
from tools import LEVINSON
import scipy as sp
import scipy.signal as sig
from spectrum import poly2lsf,poly2rc,rc2lar

# make it python3.x compatible
try:
    xrange(1)
except:
    xrange = range
def nextpow2(n):
    """Return the next power of 2 such as 2^p >= n.
        :param n: input number

        :returns: a number satisfeis  2^p >= n.
    Infinite and nan are left untouched, negative values are not allowed."""
    if numpy.any(n < 0):
        raise ValueError("n should be > 0")

    if numpy.isscalar(n):
        f, p = numpy.frexp(n)
        if f == 0.5:
            return p-1
        elif numpy.isfinite(f):
            return p
        else:
            return f
    else:
        f, p = numpy.frexp(n)
        res = f
        bet = numpy.isfinite(f)
        exa = (f == 0.5)
        res[bet] = p[bet]
        res[exa] = p[exa] - 1
        return res

def lpc2cep(lpc):
    """Compute cepstral coefficients from lpcs.

                :param lpc: Linear Prediction Coefficients features
                :returns: a numpy array as same type and size with input(lpcc feature). Each row holds 1 feature vector .

                """
    N,P = lpc.shape
    ccs = numpy.zeros([N, P])
    s=0
    i=1
    m=1
    k=1
    while i<=N:
        m=1
        while m<=P:
            s=0
            k=1
            while k<=(m-1):
                s=s+(-1*(m-k))*ccs[i-1,m-k-1]*lpc[i-1,k-1]
                k=k+1
            ccs[i-1,m-1]=(-1*lpc[i-1,m-1])+((1/(m))*s)
            m=m+1
        i=i+1
    return ccs

def lpcc(signal,samplerate=16000,winlen=0.025,winstep=0.01,winfunc='hamm',N=12):
    """Compute Linear Prediction Cepstral Coefficients features from an audio signal.

            :param signal: the audio signal from which to compute features. Should be an N*1 array
            :param samplerate: the samplerate of the signal we are working with.
            :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
            :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
            :param winfunc: the analysis window to apply to each frame. By default hamming window is applied.
            :param N: number on rc feature.
            :returns: a numpy array of size (NUMFRAMES by N) containing features. Each row holds 1 feature vector .

            """

    feat = lpcfeature(signal,samplerate=samplerate,winlen=winlen,winstep=winstep,winfunc=winfunc,N=N)
    feat=lpc2cep(feat)
    return feat

def lar(signal,samplerate=16000,winlen=0.025,winstep=0.01,winfunc='hamm',N=12):
    """Compute Log Area Ratios features from an audio signal.

            :param signal: the audio signal from which to compute features. Should be an N*1 array
            :param samplerate: the samplerate of the signal we are working with.
            :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
            :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
            :param winfunc: the analysis window to apply to each frame. By default hamming window is applied.
            :param N: number on rc feature.
            :returns: a numpy array of size (NUMFRAMES by N) containing features. Each row holds 1 feature vector .

            """

    a, e, k = lpc(signal,samplerate=samplerate,winlen=winlen,winstep=winstep,winfunc=winfunc,N=N)
    feat = numpy.zeros([a.shape[0], N])
    for i in range(a.shape[0]):
        temp2 = poly2rc(a[i],0)
        feat[i]=rc2lar(temp2)
    return feat

def lpcfeature(signal,samplerate=16000,winlen=0.025,winstep=0.01,winfunc='hamm',N=12):
    """Compute Linear Prediction Coefficients features from an audio signal.

            :param signal: the audio signal from which to compute features. Should be an N*1 array
            :param samplerate: the samplerate of the signal we are working with.
            :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
            :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
            :param winfunc: the analysis window to apply to each frame. By default hamming window is applied.
            :param N: number on rc feature.
            :returns: a numpy array of size (NUMFRAMES by N) containing features. Each row holds 1 feature vector .

            """

    a, e, k = lpc(signal,samplerate=samplerate,winlen=winlen,winstep=winstep,winfunc=winfunc,N=N)
    feat = a[:,1:]
    return feat

def lsf(signal,samplerate=16000,winlen=0.025,winstep=0.01,winfunc='hamm',N=12):
    """Compute Line Spectral Frequencies features from an audio signal.

            :param signal: the audio signal from which to compute features. Should be an N*1 array
            :param samplerate: the samplerate of the signal we are working with.
            :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
            :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
            :param winfunc: the analysis window to apply to each frame. By default hamming window is applied.
            :param N: number on rc feature.
            :returns: a numpy array of size (NUMFRAMES by N) containing features. Each row holds 1 feature vector .

            """

    a, e, k = lpc(signal,samplerate=samplerate,winlen=winlen,winstep=winstep,winfunc=winfunc,N=N)
    feat = numpy.zeros([a.shape[0], N])
    for i in range(a.shape[0]):
        feat[i] = poly2lsf(a[i])
    return feat

def rc(signal,samplerate=16000,winlen=0.025,winstep=0.01,winfunc='hamm',N=12):
    """Compute Reflection Coefficients features from an audio signal.

            :param signal: the audio signal from which to compute features. Should be an N*1 array
            :param samplerate: the samplerate of the signal we are working with.
            :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
            :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
            :param winfunc: the analysis window to apply to each frame. By default hamming window is applied.
            :param N: number on rc feature.
            :returns: a numpy array of size (NUMFRAMES by N) containing features. Each row holds 1 feature vector .

            """

    a, e, k = lpc(signal,samplerate=samplerate,winlen=winlen,winstep=winstep,winfunc=winfunc,N=N)
    feat = numpy.zeros([a.shape[0], N])
    for i in range(a.shape[0]):
        feat[i] = poly2rc(a[i],0)
    return feat

def lpc(signal,samplerate=16000,winlen=0.025,winstep=0.01,winfunc='hamm',N=None):
    """Compute Linear Prediction Coefficients  from an audio signal.

        :param signal: the audio signal from which to compute features. Should be an N*1 array
        :param samplerate: the samplerate of the signal we are working with.
        :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
        :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
        :param winfunc: the analysis window to apply to each frame. By default hamming window is applied.
        :param N: number on lpc feature.
        :returns: two numpy arrays of size (NUMFRAMES by N+1) containing features. Each row holds 1 feature vector and a numpy array of size (NUMFRAMES by 1).

        """
    x = sigproc.framesig(signal,samplerate,winlen,winstep,winfunc)
    assert x.ndim!=0 , "signal:lpc:Empty"
    if x.ndim==1:
        x = numpy.expand_dims(x, axis=0)
    [n, m] = x.shape
    if (n > 1) and (m == 1):
        x = x[:]
        [n, m] = x.shape
    if N is None:
        N = m - 1
    assert N>=0, "signal:lpc:negativeOrder"
    assert N < m, "signal:lpc:orderTooLarge', 'X must be a vector with length greater or equal to the prediction order.', 'If X is a matrix, the length of each column must be greater or equal to', 'the prediction order."
    X = numpy.fft.fft(x.T,n=int(math.pow(2,nextpow2(2*x.shape[1]-1))),axis=0)
    R = numpy.real(ifft(numpy.abs(X.T ** 2))).T
    R=R/m
    a=numpy.ndarray([])
    e=[]
    k=numpy.ndarray([])
    for i in range(R.T.shape[0]):
        A, E, K = LEVINSON(R.T[i],N,allow_singularity=True)
        if a.ndim==0:
            a=A
        else:
            a=numpy.vstack([a,A])

        if k.ndim==0:
            k=K
        else:
            k=numpy.vstack([k,K])
        e.append(E)
    e=numpy.array(e)
    a=numpy.hstack([numpy.ones([a.shape[0],1]),a])
    k = numpy.hstack([numpy.ones([k.shape[0], 1]), k])
    for i in range(x.shape[0]):
        if numpy.any(numpy.isreal(x[i])):
            a[i]=numpy.real(a[i])
    return (a,e,k)

def mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13,
         nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True,winfunc='tri'):
    """Compute MFCC features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    """
    feat, energy = fbank(signal, samplerate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph,winfunc)
    feat = numpy.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:, :numcep]
    feat = lifter(feat, ceplifter)
    if appendEnergy: feat[:, 0] = numpy.log(energy)  # replace first cepstral coefficient with log of frame energy
    return feat


def fbank(signal, samplerate=16000, winlen=0.025, winstep=0.01,
          nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,winfunc='tri'):
    """Compute Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
        second return value is the energy in each frame (total energy, unwindowed)
    """
    highfreq = highfreq or samplerate / 2
    signal = sigproc.preemphasis(signal, preemph)
    frames = sigproc.framesig(signal,samplerate,winlen, winstep,winfunc)
    pspec = sigproc.powspec(frames, nfft)
    energy = numpy.sum(pspec, 1)  # this stores the total energy in each frame
    energy = numpy.where(energy == 0, numpy.finfo(float).eps, energy)  # if energy is zero, we get problems with log

    fb = get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq)
    feat = numpy.dot(pspec, fb.T)  # compute the filterbank energies
    feat = numpy.where(feat == 0, numpy.finfo(float).eps, feat)  # if feat is zero, we get problems with log

    return feat, energy


def logfbank(signal, samplerate=16000, winlen=0.025, winstep=0.01,
             nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,winfunc='tri'):
    """Compute log Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
    """
    feat, energy = fbank(signal, samplerate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph,winfunc)
    return numpy.log(feat)


def ssc(signal,samplerate=16000,winlen=0.025,winstep=0.01,
        nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
        winfunc='tri'):
    """Compute Spectral Subband Centroid features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
    """

    highfreq= highfreq or samplerate/2
    signal = sigproc.preemphasis(signal,preemph)
    frames = sigproc.framesig(signal, samplerate,winlen, winstep, winfunc)
    pspec = sigproc.powspec(frames,nfft)
    pspec = numpy.where(pspec == 0,numpy.finfo(float).eps,pspec) # if things are all zeros we get problems

    fb = get_filterbanks(nfilt,nfft,samplerate,lowfreq,highfreq)
    feat = numpy.dot(pspec,fb.T) # compute the filterbank energies
    R = numpy.tile(numpy.linspace(1,samplerate/2,numpy.size(pspec,1)),(numpy.size(pspec,0),1))

    return numpy.dot(pspec*R,fb.T) / feat


def hz2mel(hz):
    """Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * numpy.log10(1 + hz / 700.0)

def mel2hz(mel):
    """Convert a value in Mels to Hertz

    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700 * (10 ** (mel / 2595.0) - 1)

def freq2mel(f):
    """Convert a value in Frequency to Mels

       :param f: a value in Frequency. This can also be a numpy array, conversion proceeds element-wise.
       :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
       """
    return 1125 * numpy.log(1+f/700)

def mel2freq(m):
    """Convert a value in Mels to Frequency

          :param m: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
          :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
          """
    return 700 * (numpy.exp(m/1125)-1)
def hz2bark(hz):
    """Convert a value in Hertz to Bark

             :param hz: a value in Hertz. This can also be a numpy array, conversion proceeds element-wise.
             :returns: a value in Bark. If an array was passed in, an identical sized array is returned.
             """
    return 6 * math.asinh(hz/600)

def bark2hz(ba):
    """Convert a value in Bark to Hertz

             :param ba: a value in Bark. This can also be a numpy array, conversion proceeds element-wise.
             :returns: a value in Herzt. If an array was passed in, an identical sized array is returned.
             """
    return 600 * math.sinh(ba/6)

def get_filterbanks(nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq = highfreq or samplerate / 2
    assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = numpy.linspace(lowmel, highmel, nfilt + 2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = numpy.floor((nfft + 1) * mel2hz(melpoints) / samplerate)

    fbank = numpy.zeros([nfilt, int(nfft / 2) + 1])
    for j in xrange(0, nfilt):
        for i in xrange(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in xrange(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
    return fbank


def lifter(cepstra, L=22):
    """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.

    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if L > 0:
        nframes, ncoeff = numpy.shape(cepstra)
        n = numpy.arange(ncoeff)
        lift = 1 + (L / 2) * numpy.sin(numpy.pi * n / L)
        return lift * cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra
