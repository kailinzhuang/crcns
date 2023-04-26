import numpy as np
import scipy.io.wavfile as wav
import soundsig.timefreq as ST


def timefreq(wavFileName, typeName, params=None):
    tfrep = make_tfrep(typeName, params)
    
    # read .wav file
    sampleRate, inputData = wav.read(wavFileName)
    depth = inputData.dtype.itemsize * 8
    tfrep['params']['rawSampleRate'] = sampleRate
    
   # Set default parameters where none are given
    if params is None:
        params = {}

    if typeName == 'ft':
        params = check_fields(params, ['fband', 'nstd', 'high_freq', 'low_freq', 'log'], 'Must specify params.%s for ft!\n', [125, 6, 8000, 250, 1])
        
        # Compute raw complex spectrogram
        twindow = params['nstd'] / (params['fband'] * 2.0 * np.pi)
        winLength = int(twindow * sampleRate)  # Window length in number of points
        winLength = int(winLength / 2) * 2  # Enforce even window length
        increment = int(0.001 * sampleRate)  # Sampling rate of spectrogram in number of points - set at 1 kHz
        
        # s, t0, f0, pg = GaussianSpectrum(inputData, increment, winLength, sampleRate)
        # ST.gaussian_stft returns t, freq, tf, rms
        t0, f0, s, pg = ST.gaussian_stft(inputData, sampleRate, winLength, increment)
        

        # Normalize the spectrogram within the specified frequency range
        maxIndx = np.argmax(f0 >= params['high_freq'])
        minIndx = np.argmax(f0 < params['low_freq']) + 1
        
        normedS = abs(s[minIndx:maxIndx, :])
        normedS /= np.max(normedS)
        
        # Take log-spectrogram
        if params['log']:
            DBNOISE = 80
            normedS = np.maximum(0, 20 * np.log10(normedS) + DBNOISE)
        
        # Set tfrep values
        fstep = f0[1] - f0[0]
        tfrep["t"] = t0
        tfrep["f"] = f0[minIndx:maxIndx + 1: int(fstep)]
        tfrep["spec"] = normedS.tolist()

    elif typeName == 'wavelet':
        # Do nothing...
        pass

    elif typeName == 'lyons':
        pass

    return tfrep
    

def make_tfrep(typeName, params=None):
    """Create a time-frequency representation structure.
    
    Args:
        typeName (str): Type of time-frequency representation. Must be one of 'ft', 'wavelet', or 'lyons'.
        params (dict): Parameters to assign to tfrep (optional). If not given, default values will be specified for type.
    
    Returns:
        tfrep (dict): The time-frequency structure, for use with display_tfrep.
    """
    # Type checking
    allowed_types = ['ft', 'wavelet', 'lyons']
    if typeName not in allowed_types:
        raise ValueError(f'Unknown time-frequency representation type: {typeName}')
    
    # Create structure
    tfrep = {}
    tfrep['type'] = typeName
    tfrep['t'] = []
    tfrep['f'] = []
    tfrep['spec'] = []
    
    # Set default parameters where none are given
    if params is None:
        params = {}
    
    if typeName == 'ft':
        params = check_fields(params, ['fband', 'nstd', 'high_freq', 'low_freq', 'log'], 
                              'Must specify params.%s for ft!\n', [125, 6, 8000, 250, 1])
    elif typeName == 'wavelet':
    # Do nothing...
        pass
    elif typeName == 'lyons':
    # Do nothing...
        pass
    tfrep['params'] = params
    
    return tfrep



def check_fields(struct_instance, required_fields, message_template, default_values=None):
    has_defaults = default_values is not None
    if has_defaults and len(required_fields) != len(default_values):
        raise ValueError('The # of default values has to match the # of required fields!')

    for field_name in required_fields:
        if field_name not in struct_instance:
            if not has_defaults:
                raise ValueError(message_template % field_name)
            else:
                default_value = default_values[required_fields.index(field_name)]
                # print(f'check_fields: setting {field_name}={default_value}')
                struct_instance[field_name] = default_value

    return struct_instance
