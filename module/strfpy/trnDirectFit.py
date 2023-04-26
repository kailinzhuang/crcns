import tempfile
import os
import numpy as np
from scipy.signal.windows import dpss
from .strfSetup import strflab2DS
from .DirectFit import direct_fit

def trnDirectFit(modelParams=None, datIdx=None, options=None, globalDat=None, *args, **kwargs):
    """
    Trains a direct fit model and sets the model parameters.

    Args:
    - modelParams: dictionary containing model parameters
    - datIdx: indices of the data to be used for training
    - options: dictionary containing options for training
    - *args: optional additional arguments

    Returns:
    - modelParams: updated dictionary of model parameters
    - options: updated dictionary of options
    """

    ## set default parameters and return if no arguments are passed
    if len(args) == 0:
        options = {}
        options['funcName'] = 'trnDirectFit'
        options['tolerances'] = [0.100, 0.050, 0.010, 0.005, 1e-03, 5e-04, 1e-04, 5e-05]
        options['sparsenesses'] = [0, 1, 2, 6]
        options['separable'] = 0
        options['timeVaryingPSTH'] = 0
        options['timeVaryingPSTHTau'] = 41
        options['stimSampleRate'] = 1000
        options['respSampleRate'] = 1000
        options['infoFreqCutoff'] = 100
        options['infoWindowSize'] = 0.500

        tempDir = tempfile.gettempdir()
        options['outputDir'] = tempDir

        modelParams = options
        return modelParams, options

    if modelParams['type'] != 'lin' or modelParams['outputNL'] != 'linear':
        raise ValueError('trnDirectFit only works for linear models with no output nonlinearity!')
        
    if options['respSampleRate'] != options['stimSampleRate']:
        raise ValueError('trnDirectFit: Stimulus and response sampling rate must be equal!')

    global globDat
    
    globDat = globalDat
    
    os.makedirs(options['outputDir'], exist_ok=True)
    
    # convert strflab's stim/response data format to direct fit's data format
    DS = strflab2DS(globDat['stim'], globDat['resp'], globDat['groupIdx'], options['outputDir'])
    
    # set up direct fit parameters
    params = {
        'DS': DS,
        'NBAND': globDat['stim'].shape[1],
        'Tol_val': options['tolerances'],
        'setSep': options['separable'],
        'TimeLagUnit': 'frame',
        'timevary_PSTH': 0,
        'smooth_rt': options['timeVaryingPSTHTau'],
        'ampsamprate': options['stimSampleRate'],
        'respsamprate': options['respSampleRate'],
        'outputPath': options['outputDir'],
        'use_alien_space': 0,
        'alien_space_file': '',
        'TimeLag': int(np.ceil(np.max(np.abs(modelParams['delays']))))
    }
    
    # run direct fit
    strfFiles = direct_fit(params)
    
    # get computed stim and response means
    svars = np.load(os.path.join(options['outputDir'], 'stim_avg.npz'), allow_pickle=True)
    stimAvg = svars['stim_avg']
    respAvg = svars['constmeanrate']
    tvRespAvg = svars['Avg_psth']
    
    numSamples = len(DS)
    
    # compute some indices to use later
    halfIndx = params['TimeLag'] + 1
    startIndx = halfIndx + round(np.min(modelParams['delays']))
    endIndx = halfIndx + round(np.max(modelParams['delays']))
    strfRng = range(startIndx, endIndx+1)
    
    # subtract mean off of stimulus (because direct fit does this)
    for k in range(datIdx['stim'].shape[0]):
        datIdx['stim'][k,:] -= stimAvg
    
    # compute information values for each set of jackknifed strfs per tolerance value
    print('Finding best STRF by computing info values across sparseness and tolerance values...')
    bestInfoVal = -1
    bestStrf = -1
    bestTol = -1
    bestSparseness = -1
    for k in range(len(strfFiles)):    # for each tolerance value
        svars = np.load(strfFiles[k], allow_pickle=True).item()
        
        strfsJN = svars['STRFJN_Cell']
        strfsJN_std = svars['STRFJNstd_Cell']
        strfMean = svars['STRF_Cell']
        strfStdMean = np.mean(strfsJN_std, axis=2)

        spvals = options['sparsenesses']

    for q in range(len(spvals)):
        # smooth the strf by masking it with a sigmoid-like mask, scaled by the # of deviations specified by the sparseness parameter
        smoothedMeanStrf = df_fast_filter_filter(strfMean, strfStdMean, spvals[q])
        smoothedMeanStrfToUse = smoothedMeanStrf[:, strfRng]

        # the following loop goes through each jacknifed strf for the given tolerance value, smooths it with the sparseness parameter, and then predicts response to it's corresponding held-out stimulus.
        # the coherence and information values are computed and recorded, then the average info value is used to judge the goodness-of-fit for smoothedMeanStrfToUse  
        infoSum = 0
        numJNStrfs = numSamples
        for p in range(numJNStrfs):
            # jacknifed STRF p (strfsJN[:, :, p]) was constructed by holding out stim/resp pair p
            smoothedMeanStrfJN = df_fast_filter_filter(strfsJN[:, :, p], strfsJN_std[:, :, p], spvals[q])
            strfToUse = smoothedMeanStrfJN[:, strfRng]
            
            srRange = np.where(globDat['groupIdx'] == p)[0]
            stim = globDat['stim'][srRange, :]
            rresp = globDat['resp'][srRange]
            gindx = np.ones((1, stim.shape[0]))

            #compute the prediction for the held out stimulus
            mresp = conv_strf(stim, modelParams['delays'], strfToUse, gindx)

            #add the mean back to the PSTH if necessary 
            # Why is this not in conv_strf???
            if not options['timeVaryingPSTH']:
                mresp = mresp + respAvg
            else:
                mresp = mresp + tvRespAvg[p, :len(mresp)]

            #compute coherence and info across pairs
            cStruct = compute_coherence_mean(rv(mresp), rv(rresp), options['respSampleRate'], options['infoFreqCutoff'], options['infoWindowSize'])
            infoSum = infoSum + cStruct['info']
        
        avgInfo = infoSum / numJNStrfs

        print(f"Tolerance={options['tolerances'][k]}, Sparseness={spvals[q]}, Avg. Prediction Info={avgInfo}")

        # did this sparseness do better?
        if avgInfo > bestInfoVal:
            bestTol = options['tolerances'][k]
            bestSparseness = spvals[q]
            bestInfoVal = avgInfo
            bestStrf = smoothedMeanStrfToUse
                

        ## get best strf
    print('Best STRF found at tol=%f, sparseness=%d, info=%f bits\n' % (bestTol, bestSparseness, bestInfoVal))
    
    modelParams['w1'] = bestStrf
    
    if not options['timeVaryingPSTH']:
        modelParams['b1'] = respAvg
    else:
        modelParams['b1'] = tvRespAvg[p, :mresp.shape[1]]

     
    return modelParams, options
    




def df_fast_filter_filter(forward, forwardJN_std, nstd):
    # smooths out the filter for displaying or calculation purposes.
    # Faster than filter_filter, but less fancy.
    # Scales the filter everywhere by a sigmoid in forward/forwardJN_std, with
    # inflection point at nstd, and a dynamic range from nstd - .5 to nstd + .5
    
    if nstd > 0:
        epsilon = 10**-8 # To prevent division by 0.
        factor = (1 + np.tanh(2*(np.abs(forward)-np.abs(nstd*forwardJN_std))/(epsilon + np.abs(forwardJN_std))))/2
        s_forward = factor * forward
    else:
        s_forward = forward
    
    return s_forward



def conv_strf(allstim, delays, strf, groupIndex):
    nDatasets = len(np.unique(groupIndex))
    timeLen = allstim.shape[0]
    a = np.zeros((timeLen, 1))
    
    for k in range(nDatasets):
        rng = np.where(groupIndex == k+1)[0]
        soff = rng[0]
        stim = allstim[rng, :]
        for ti in range(len(delays)):
            at = np.dot(stim, strf[:, ti])

            thisshift = delays[ti]
            if thisshift >= 0:
                a[soff+thisshift+1:] = a[soff+thisshift+1:] + at[:-thisshift]
            else:
                offset = thisshift % timeLen
                a[soff:offset] = a[soff:offset] + at[-thisshift:]
    
    return a.T[0]



def compute_coherence_mean(modelResponse, psth, sampleRate, freqCutoff=-1, windowSize=0.500):
    import numpy as np
    
    # put psths in matrix for mtchd_JN
    if len(modelResponse) != len(psth):
        minLen = min(len(modelResponse), len(psth))
        modelResponse = modelResponse[:minLen]
        psth = psth[:minLen]
    x = np.column_stack([modelResponse, psth])
    
    # compute # of time bins per FFT segment
    minFreq = round(1 / windowSize)
    numTimeBin = round(sampleRate * windowSize)
    
    # get default parameter values
    vargs = [x, numTimeBin, sampleRate]
    x, nFFT, Fs, WinLength, nOverlap, NW, Detrend, nTapers = df_mtparam(*vargs)
    
    # compute jacknifed coherence
    y, fpxy, cxyo, cxyo_u, cxyo_l, stP = df_mtchd_JN(x, nFFT, Fs, WinLength, nOverlap, NW, Detrend, nTapers)
    
    # normalize coherencies
    cStruct = {}
    cStruct['f'] = fpxy
    cStruct['c'] = cxyo[:, 0, 1]**2
    cStruct['cUpper'] = cxyo_u[:, 0, 1]**2
    
    clo = cxyo_l[:, 0, 1]
    closgn = np.sign(np.real(clo))
    cStruct['cLower'] = (clo**2) * closgn
    
    # restrict frequencies analyzed to the requested cutoff and minimum frequency given the window size
    if freqCutoff != -1:
        indx = np.where(cStruct['f'] < freqCutoff)[0]
        eindx = max(indx)
        indx = np.arange(eindx)
        
        cStruct['f'] = cStruct['f'][indx]
        cStruct['c'] = cStruct['c'][indx]
        cStruct['cUpper'] = cStruct['cUpper'][indx]
        cStruct['cLower'] = cStruct['cLower'][indx]
    
    if minFreq > 0:
        indx = np.where(cStruct['f'] >= minFreq)[0]
        sindx = min(indx)
        cStruct['f'] = cStruct['f'][sindx:]
        cStruct['c'] = cStruct['c'][sindx:]
        cStruct['cUpper'] = cStruct['cUpper'][sindx:]
        cStruct['cLower'] = cStruct['cLower'][sindx:]
    
    # compute information by integrating log of 1 - coherence
    df = cStruct['f'][1] - cStruct['f'][0]
    cStruct['minFreq'] = minFreq
    cStruct['info'] = -df * np.sum(np.log2(1 - cStruct['c']))
    cStruct['infoUpper'] = -df * np.sum(np.log2(1 - cStruct['cUpper']))
    cStruct['infoLower'] = -df * np.sum(np.log2(1 - cStruct['cLower']))
    
    return cStruct



def rv(a):
    b = a
    sz = np.shape(a)
    isvect = (sz[0] == 1) or (sz[1] == 1)
    if (isvect):
        if (sz[0] == 1):
            b = a.T
    return b



def df_mtparam(P):
    nargs = len(P)

    x = P[0]
    if nargs < 2 or P[1] is None:
        nFFT = 1024
    else:
        nFFT = P[1]
    if nargs < 3 or P[2] is None:
        Fs = 2
    else:
        Fs = P[2]
    if nargs < 4 or P[3] is None:
        WinLength = nFFT
    else:
        WinLength = P[3]
    if nargs < 5 or P[4] is None:
        nOverlap = WinLength // 2
    else:
        nOverlap = P[4]
    if nargs < 6 or P[5] is None:
        NW = 3
    else:
        NW = P[6]
    if nargs < 7 or P[6] is None:
        Detrend = ''
    else:
        Detrend = P[6]
    if nargs < 8 or P[7] is None:
        nTapers = 2 * NW - 1
    else:
        nTapers = P[7]

    # Now do some computations that are common to all spectrogram functions
    winstep = WinLength - nOverlap

    nChannels = x.shape[1]
    nSamples = x.shape[0]

    # check for column vector input
    if nSamples == 1:
        x = x.T
        nSamples = x.shape[0]
        nChannels = 1

    # calculate number of FFTChunks per channel
    nFFTChunks = round(((nSamples - WinLength) / winstep))
    # turn this into time, using the sample frequency
    t = winstep * np.arange(nFFTChunks) / Fs

    # set up f and t arrays
    if np.all(np.isreal(x)):
        # x purely real
        if nFFT % 2:
            # nfft odd
            select = np.arange(1, (nFFT + 1) // 2)
        else:
            select = np.arange(1, nFFT // 2 + 1)
        nFreqBins = len(select)
    else:
        select = np.arange(1, nFFT + 1)
        nFreqBins = nFFT

    f = (select - 1) * Fs / nFFT

    return x, nFFT, Fs, WinLength, nOverlap, NW, Detrend, nTapers, nChannels, nSamples, nFFTChunks, winstep, select, nFreqBins, f, t





def df_mtchd_JN(x, nFFT=1024, Fs=2, WinLength=None, nOverlap=None, NW=3, Detrend=None, nTapers=None):
    if WinLength is None:
        WinLength = nFFT
    if nOverlap is None:
        nOverlap = WinLength // 2
    if nTapers is None:
        nTapers = 2 * NW - 1

    WinLength = int(WinLength)
    nOverlap = int(nOverlap)
    nFFT = int(nFFT)

    nChannels = x.shape[1]
    nSamples = x.shape[0]

    # check for column vector input
    if nSamples == 1:
        x = x.T
        nSamples = x.shape[0]
        nChannels = 1

    # calculate number of FFTChunks per channel
    winstep = WinLength - nOverlap
    nFFTChunks = ((nSamples - WinLength) // winstep)
    # turn this into time, using the sample frequency
    t = winstep * np.arange(nFFTChunks) / Fs

    # calculate Slepian sequences. Tapers is a matrix of size [WinLength, nTapers]

    # [JN,y,stP] = make_slepian(x,WinLength,NW,nTapers,nChannels,nFFTChunks,nFFT,Detrend,winstep);
    # allocate memory now to avoid nasty surprises later
    stP = np.zeros((nFFT, nChannels, nChannels))
    varP = np.zeros((nFFT, nChannels, nChannels))
    Tapers, V = dpss(WinLength, NW, nTapers, 'calc')
    Periodogram = np.zeros((nFFT, nTapers, nChannels), dtype=complex)  # intermediate FFTs
    Temp1 = np.zeros((nFFT, nTapers), dtype=complex)  # Temps are particular psd or csd values for a frequency and taper
    Temp2 = np.zeros((nFFT, nTapers), dtype=complex)
    Temp3 = np.zeros((nFFT, nTapers), dtype=complex)
    eJ = np.zeros((nFFT,), dtype=complex)
    JN = np.zeros((nFFTChunks, nFFT, nChannels, nChannels), dtype=complex)
    # jackknifed cross-spectral-densities or csd. Note: JN(.,.,1,1) is
    # the power-spectral-density of time series 1 and JN(.,.,2,2) is the
    # psd of time series 2. Half-way through this code JN(.,.,1,2)
    # ceases to be the csd of 1 and 2 and becomes the abs coherency of 1
    # and 2.
    y = np.zeros((nFFT, nChannels, nChannels), dtype=complex)  # output array for csd
    Py = np.zeros((nFFT, nChannels, nChannels))  # output array for psd's

    # New super duper vectorized alogirthm
    # compute tapered periodogram with FFT
    # This involves lots of wrangling with multidimensional arrays.

    TaperingArray = np.tile(Tapers[:, np.newaxis, :], (1, nChannels, 1))

    for j in range(nFFTChunks):
        start_idx = (j-1)*winstep
        end_idx = start_idx + WinLength
        Segment = x[start_idx:end_idx, :]
        if Detrend is not None:
            Segment = np.apply_along_axis(lambda x: np.squeeze(np.detrend(x, Detrend)), axis=0, arr=Segment)
        SegmentsArray = np.transpose(np.tile(Segment, (1, nTapers, 1)), (0, 2, 1))
        TaperedSegments = TaperingArray * SegmentsArray

        Periodogram = np.fft.fft(TaperedSegments, n=nFFT, axis=0)

        for Ch1 in range(nChannels):
            for Ch2 in range(Ch1, nChannels):
                Temp1 = Periodogram[:, :, Ch1]
                Temp2 = np.conj(Periodogram[:, :, Ch2])
                Temp3 = Temp1 * Temp2

                # eJ and eJ2 are the sum over all the tapers.
                eJ = np.sum(Temp3, axis=1) / nTapers
                JN[j, :, Ch1, Ch2] = eJ  # Here it is just the cross-power for one particular chunk.
                y[:, Ch1, Ch2] += eJ  # y is the sum of the cross-power

    # now fill other half of matrix with complex conjugate
    for Ch1 in range(nChannels):
        for Ch2 in range(Ch1+1, nChannels): # don't compute cross-spectra twice
            y[:, Ch2, Ch1] = y[:, Ch1, Ch2]
            Py[:, Ch1, Ch2] = np.arctanh(abs(y[:, Ch1, Ch2] / np.sqrt(abs(y[:, Ch1, Ch1]) * abs(y[:, Ch2, Ch2]))))

    for j in range(nFFTChunks):
        JN[j, :, :, :] = abs(y - np.squeeze(JN[j, :, :, :])) # This is where it becomes the JN quantity (the delete one)
        for Ch1 in range(nChannels):
            for Ch2 in range(Ch1+1, nChannels):
                # Calculate the transformed coherence
                JN[j, :, Ch1, Ch2] = np.arctanh(np.real(JN[j, :, Ch1, Ch2]) / np.sqrt(abs(JN[j, :, Ch1, Ch1]) * abs(JN[j, :, Ch2, Ch2])))
                # Obtain the pseudo values
                JN[j, :, Ch1, Ch2] = nFFTChunks * Py[:, Ch1, Ch2].T - (nFFTChunks-1) * np.squeeze(JN[j, :, Ch1, Ch2])

    # upper and lower bounds will be 2 standard deviations away
    stP = np.sqrt(varP)

    Pupper = np.tanh(meanP + 2*stP)
    Plower = np.tanh(meanP - 2*stP)
    meanP = np.tanh(meanP)

    # set up f array
    if not np.any(np.any(np.imag(x))):
        # x purely real
        if nFFT % 2 == 1:  # nfft odd
            select = np.arange(1, (nFFT + 1) // 2 + 1)
        else:
            select = np.arange(1, nFFT // 2 + 2)
        meanP = meanP[select, :, :]
        Pupper = Pupper[select, :, :]
        Plower = Plower[select, :, :]
        y = y[select, :, :]
    else:
        select = np.arange(1, nFFT + 1)

    fo = (select - 1) * Fs / nFFT

    ###### did not translate
    ######
    # if nargout == 0
    #     % take abs, and plot results
    #     newplot;
    #     for Ch1=1:nChannels, 
    #         for Ch2 = 1:nChannels
    #             subplot(nChannels, nChannels, Ch1 + (Ch2-1)*nChannels);
    #             plot(f,20*log10(abs(y(:,Ch1,Ch2))+eps));
    #             grid on;
    #             if(Ch1==Ch2) 
    #                 ylabel('psd (dB)'); 
    #             else 
    #                 ylabel('csd (dB)'); 
    #             end;
    #             xlabel('Frequency');
    #         end
    #     end
    # end

    return y, fo, meanP, Pupper, Plower, stP

