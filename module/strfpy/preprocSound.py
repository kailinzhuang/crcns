import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import convolve
from .timeFreq import timefreq
import pandas as pd



def preprocess_sound(raw_stim_files, raw_resp_files, preprocess_type='ft', stim_params=None,
                     output_dir=None, stim_output_pattern='preprocessed_stim_%d.mat',
                     resp_output_pattern='preprocessed_resp_%d.mat'):
    
    if len(raw_stim_files) != len(raw_resp_files):
        raise ValueError('# of stim and response files must be the same!')
    
    if preprocess_type not in ['ft', 'wavelet', 'lyons']:
        raise ValueError('Unknown time-frequency representation type: %s' % preprocess_type)
    
    if stim_params is None:
        stim_params = {}
    
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'temp')
    os.makedirs(output_dir, exist_ok=True)
    

    
    max_stim_len = -1
    max_resp_len = -1
    n_stim_channels = -1
    stim_sample_rate = 1000
    resp_sample_rate = 1000
    pair_count = len(raw_stim_files)
    srData = {}
    datasets = [None] * pair_count

    # preprocess each stimulus and response
    for k in range(pair_count):
        ds = {}

        # preprocess stimulus
        stim_output_fname = os.path.join(output_dir, stim_output_pattern % (k + 1))

        if os.path.isfile(stim_output_fname):
            # use cached preprocessed stimulus
            #print(f'Using cached preprocessed stimulus from {stim_output_fname}')
            ds['stim'] = loadmat(stim_output_fname)['stim']
        else:
            wav_file_name = raw_stim_files[k]
            tfrep = timefreq(wav_file_name, preprocess_type, stim_params)
            stim = {
                'type': 'tfrep',
                'rawFile': wav_file_name,
                'tfrep': tfrep,
                'rawSampleRate': tfrep.params.rawSampleRate,
                'sampleRate': stim_sample_rate,
                'stimLength': tfrep.spec.shape[1] / stim_sample_rate
            }
            np.savez(stim_output_fname, stim=stim)
            ds['stim'] = stim

        # preprocess response
        resp_output_fname = os.path.join(output_dir, resp_output_pattern % (k + 1))
        if os.path.isfile(resp_output_fname):
            # use cached preprocessed response
            #print(f'Using cached preprocessed response from {resp_output_fname}')
            ds['resp'] = loadmat(resp_output_fname)['resp']
        else:
            spike_trials = read_spikes_from_file(raw_resp_files[k])
            resp = preprocess_response(spike_trials, ds['stim']['stimLength'], resp_sample_rate)
            np.savez(resp_output_fname, resp=resp)
            ds['resp'] = resp


        # update max sizes
        if n_stim_channels == -1:
            n_stim_channels = pd.DataFrame(ds['stim'].flatten()['tfrep'][0].flatten())['spec'][0].shape[0]
        if ds['stim'].flatten()['tfrep'][0].flatten()['spec'][0].shape[1] > max_stim_len:
            max_stim_len = pd.DataFrame(ds['stim'].flatten()['tfrep'][0].flatten())['spec'][0].shape[1]
        if len(ds['resp']['psth']) > max_resp_len:
            max_resp_len = len(ds['resp']['psth'])

        datasets[k] = ds
        
    # set dataset-wide values
    srData = {
        'stimSampleRate': stim_sample_rate,
        'respSampleRate': resp_sample_rate,
        'nStimChannels': n_stim_channels,
        'datasets': datasets
    }

    # return srData
    # compute averages
    stim_avg, resp_avg, tv_resp_avg = compute_srdata_means(srData)
    srData['stimAvg'] = stim_avg
    srData['respAvg'] = resp_avg
    srData['tvRespAvg'] = tv_resp_avg
    srData['type'] = preprocess_type

    return srData



def read_spikes_from_file(file_name):
    spike_times = np.loadtxt(file_name, delimiter=' ')
    spike_trials = []
    for i in range(spike_times.shape[0]):
        trial = spike_times[i, spike_times[i, :] > 0]
        spike_trials.append(trial)
    return spike_trials



def preprocess_response(spikeTrials, stimLength, sampleRate):
    nSpikeTrials = len(spikeTrials)
    spikeIndicies = []
    for stimes in spikeTrials:
        # turn spike times (ms) into indexes at response sample rate
        stimes = np.round(stimes*1e-3 * sampleRate).astype(int)
        # remove excess zeros
        stimes = stimes[stimes > 0]
        spikeIndicies.append(stimes)
    psth = make_psth(spikeTrials, stimLength*1e3, 1)
    resp = {
        'type': 'psth',
        'sampleRate': sampleRate,
        'rawSpikeTimes': spikeTrials,
        'rawSpikeIndicies': spikeIndicies,
        'psth': psth
    }
    return resp


def make_psth(spikeTrials, stimdur, binsize):
    nbins = round(stimdur / binsize)
    psth = np.zeros(nbins)

    ntrials = len(spikeTrials)

    maxIndx = round(stimdur / binsize)

    for k in range(ntrials):
        stimes = spikeTrials[k]
        indx = np.logical_and(stimes.any() > 0, stimes.any() < stimdur)
        indx = indx[0]
        stimes = stimes[0][0]
        stimes = stimes[indx]
        sindxs = np.round(stimes / binsize).astype(int) + 1

        sindxs[sindxs == 0] = 1
        sindxs[sindxs > maxIndx] = maxIndx

        psth[sindxs - 1] += 1

    psth /= ntrials
    return psth

def compute_srdata_means(srData):

    pairCount = len(srData['datasets'])

    # get max response length
    maxRespLen = -1
    for k in range(pairCount):
        ds = srData['datasets'][k]
        if len(srData['datasets'][1]['resp']['psth'].flatten()[0].flatten()) > maxRespLen:
            maxRespLen = len(srData['datasets'][1]['resp']['psth'].flatten()[0].flatten())

    # compute stim and response averages
    stimSum = np.zeros((srData['nStimChannels'], ))
    stimCountSum = 0
    respSum = np.zeros((1, maxRespLen))
    meanSum = 0
    tvRespCount = np.zeros((pairCount, maxRespLen))

    # first compute all the sums
    for k in range(pairCount):
        ds = srData['datasets'][k]
        stimSum += np.sum(ds['stim'].flatten()['tfrep'][0].flatten()['spec'][0], axis=1)
        stimCountSum += ds['stim'].flatten()['tfrep'][0].flatten()['spec'][0].shape[1]

        rlen = maxRespLen - len(srData['datasets'][1]['resp']['psth'].flatten()[0].flatten())

        nresp = np.append(ds['resp']['psth'].flatten()[0], np.zeros((1, rlen)))

        respSum = respSum + nresp[0]


        tvIndx = np.arange(len(srData['datasets'][1]['resp']['psth'].flatten()[0].flatten()))
        tvRespCount[k, tvIndx] = 1


        meanSum = meanSum + ds['resp']['psth'].flatten()[0].mean()

    # construct the time-varying mean for the response. each row of the tv-mean is the average PSTH (across pairs)
    # computed with the PSTH of that row index left out
    tvRespCountSum = np.sum(tvRespCount, axis=0)
    tvRespAvg = np.zeros((pairCount, maxRespLen))
    smoothWindowTau = 41
    hwin = np.hanning(smoothWindowTau)
    hwin = hwin / np.sum(hwin)
    halfTau = smoothWindowTau // 2
    coff = smoothWindowTau % 2
    for k in range(pairCount):
        ds = srData['datasets'][k]
        rlen = maxRespLen - len(srData['datasets'][1]['resp']['psth'].flatten()[0].flatten())
        nresp = np.append(ds['resp']['psth'].flatten()[0], np.zeros((1, rlen)))

        # subtract off this pair's PSTH, construct mean
        tvcnts = tvRespCountSum - tvRespCount[k, :]
        tvcnts[tvcnts < 1] = 1

        tvRespAvg[k, :] = (respSum - nresp[0]) / tvcnts

        # smooth with hanning window
        pprod = convolve(tvRespAvg[k, :], hwin, mode='full')
        sindx = halfTau+coff
        eindx = round(len(pprod)-halfTau) + 1
        tvRespAvg[k, :] = pprod[sindx:eindx]

    stimAvg = stimSum / stimCountSum
    respAvg = meanSum / pairCount

    return stimAvg, respAvg, tvRespAvg


def split_psth(spikeTrials, stimLengthMs):
    spikeTrials = spikeTrials.flatten()[0]
    halfSize = round(len(spikeTrials)/2)
    spikeTrials1 = [None]*halfSize
    spikeTrials2 = [None]*halfSize


    for j, trial in enumerate(spikeTrials):
        indx = int((j + 1) // 2)
        if j%2 == 0:
            spikeTrials1[indx-1] = trial
        else:
            spikeTrials2[indx-1] = trial
            
    psth = make_psth(spikeTrials, stimLengthMs, 1)
    psth1 = make_psth(spikeTrials1, stimLengthMs, 1)
    psth2 = make_psth(spikeTrials2, stimLengthMs, 1)

    psthdata = {'psth': psth, 'psth_half1': psth1, 'psth_half2': psth2}
    return psthdata