# ChatGPT convert of 
# trnDirectFit.m

import os
from tempfile import gettempdir
import numpy as np
from numpy import ceil, max, abs, round, arange, linspace, load
from .direct_fit import direct_fit
from .strflab2DS import strflab2DS

def trnDirectFit(modelParams, datIdx, options=None, **kwargs):

    # set default parameters and return if no arguments are passed
    if options is None:
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

        tempDir = gettempdir()
        options['outputDir'] = tempDir

        modelParams = options

        return modelParams, options

    if modelParams['type'] != 'lin' or modelParams['outputNL'] != 'linear':
        raise ValueError('trnDirectFit only works for linear models with no output nonlinearity!')

    if options['respSampleRate'] != options['stimSampleRate']:
        raise ValueError('trnDirectFit: Stimulus and response sampling rate must be equal!')

    print('Writing temp direct fit output to %s' % options['outputDir'])
    os.makedirs(options['outputDir'], exist_ok=True)

    # convert strflab's stim/response data format to direct fit's data format
    DS = strflab2DS(globDat.stim, globDat.resp, globDat.groupIdx, options['outputDir'])

    # set up direct fit parameters
    params = {
        'DS': DS,
        'NBAND': globDat.stim.shape[1],
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
        'TimeLag': ceil(max(abs(modelParams['delays']))),
    }

    # run direct fit
    strfFiles = direct_fit(params)

    # get computed stim and response means
    svars = load(os.path.join(options['outputDir'], 'stim_avg.mat'))
    stimAvg = svars['stim_avg']
    respAvg = svars['constmeanrate']
    tvRespAvg = svars['Avg_psth']
    del svars

    numSamples = len(DS)

    # compute some indices to use later
    halfIndx = params['TimeLag'] + 1  # This is the point corresponding to zero
    startIndx = halfIndx + round(min(modelParams['delays']))
    endIndx = halfIndx + round(max(modelParams['delays']))
    strfRng = startIndx:endIndx


    # Subtract mean off of stimulus
    for k in range(globDat.stim.shape[0]):
        globDat.stim[k, :] = globDat.stim[k, :] - stimAvg

    # Compute information values for each set of jacknifed strfs per tolerance value
    print('Finding best STRF by computing info values across sparseness and tolerance values...\n')

    bestInfoVal = -1
    bestStrf = -1
    bestTol = -1

    bestSparseness = -1
    for k in range(len(strfFiles)):  # for each tolerance value
        svars = np.load(strfFils, preprocessType, stimParams, outputDir, stimOutputPattern, respOutputPattern)es[k], allow_pickle=True)

        # strfsJN is an MxPxT matrix, where M=# of channels, P=# of STRF
        # delays, T = # of stim/response pairs, each element strfsJN[:, :, k] is a STRF
        # constructed from fitting all but pair k to the data.
        strfsJN = svars['STRFJN_Cell']

        # strfsJN_std is also an MxPxT matrix. Element strfsJN_std[:, :, k]
        # is the STRF standard deviation across the set of all jacknifed
        # STRFs excluding the kth STRF
        strfsJN_std = svars['STRFJNstd_Cell']

        # strfMean is the mean across all jacknifed STRFs for a given
        # tolerance
        strfMean = svars['STRF_Cell']

        # strfStdMean is the average standard deviation across all jacknifed
        # STRFs for a given tolerance
        strfStdMean = np.mean(strfsJN_std, axis=2)

        spvals = options.sparsenesses
        for q in range(len(spvals)):  # for each sparseness value
            # smooth the strf by masking it with a sigmoid-like mask,
            # scaled by the # of deviations specified by the sparseness
            # parameter
            smoothedMeanStrf = df_fast_filter_filter(strfMean, strfStdMean, spvals[q])
            smoothedMeanStrfToUse = smoothedMeanStrf[:, strfRng]

            # the following loop goes through each jacknifed strf for the given tolerance
            # value, smooths it with the sparseness parameter, and then
            # predicts response to it's corresponding held-out stimulus.
            # the coherence and information values are computed and recorded,
            # then the average info value is used to judge the goodness-of-fit
            # for smoothedMeanStrfToUse
            infoSum = 0  # this will be used to compute the average
            numJNStrfs = numSamples
            for p in range(numJNStrfs):
                # jacknifed STRF p (strfsJN[:, :, p]) was constructed by holding out stim/resp pair p
                smoothedMeanStrfJN = df_fast_filter_filter(strfsJN[:, :, p], strfsJN_std[:, :, p], spvals[q])
                strfToUse = smoothedMeanStrfJN[:, strfRng]

                # get the held-out stimulus
                srRange = np.where(globDat.groupIdx == p)
                stim = globDat.stim[srRange, :]
                rresp = globDat.resp[srRange]
                gindx = np.ones((1, stim.shape[0]))

                # compute the prediction for the held out stimulus
                mresp = conv_strf(stim, modelParams.delays, strfToUse, gindx)
                            
                # add the mean back to the PSTH if necessary 
                # Why is this not in conv_strf???
                if not options.timeVaryingPSTH:
                    mresp = mresp + respAvg
                else:
                    mresp = mresp + tvRespAvg[p, :len(mresp)]
                            
                # compute coherence and info across pairs
                cStruct = compute_coherence_mean(rv(mresp), rv(rresp), options.respSampleRate, options.infoFreqCutoff, options.infoWindowSize)
                infoSum = infoSum + cStruct.info

            avgInfo = infoSum / numJNStrfs

            print(f'Tolerance={options.tolerances[k]}, Sparseness={spvals[q]}, Avg. Prediction Info={avgInfo}')

            # did this sparsenes, preprocessType, stimParams, outputDir, stimOutputPattern, respOutputPattern)ss do better?
            if avgInfo > bestInfoVal:
                bestTol = options.tolerances[k]
                bestSparseness = spvals[q]
                bestInfoVal = avgInfo
                bestStrf = smoothedMeanStrfToUse

    print(f'Best STRF found at tol={bestTol}, sparseness={bestSparseness}, info={bestInfoVal} bits')

    modelParams.w1 = bestStrf
    if not options.timeVaryingPSTH:
        modelParams.b1 = respAvg
    else:
        modelParams.b1 = tvRespAvg[p, :len(mresp)]


