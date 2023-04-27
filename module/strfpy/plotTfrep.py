import matplotlib.pyplot as plt
import numpy as np

def plot_tfrep(tfrep, ax=None):

    if tfrep['type'] == 'strfpak':
        tfrep['type'] = 'ft'

    ## type checking
    allowedTypes = ['ft', 'wavelet', 'lyons']
    if tfrep['type'] not in allowedTypes:
        raise ValueError(f"Cannot display time-frequency representation of type {tfrep['type']}!")

    ## plot stuff common to all types
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(tfrep['spec'], origin='lower', aspect='auto', extent=[tfrep['t'][0], tfrep['t'][-1], tfrep['f'][0], tfrep['f'][-1]])
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')

    ## plot type-specific stuff
    if tfrep['type'] == 'ft':
        DBNOISE = 50
        maxB = np.max(tfrep['spec'])
        minB = maxB - DBNOISE

        im.set_clim(minB, maxB)

        cmap = spec_cmap()
        im.set_cmap(cmap)

    elif tfrep['type'] == 'wavelet':
        pass

    elif tfrep['type'] == 'lyons':
        im.set_cmap('viridis')

    return ax




def spec_cmap():
    cmap = np.zeros((64, 3))
    cmap[0, 2] = 1.0

    for ib in range(21):
        cmap[ib+1, 0] = (31+(ib-1)*(12/20))/60
        cmap[ib+1, 1] = ib/21
        cmap[ib+1, 2] = 1.0

    for ig in range(21):
        cmap[ig+ib+1, 0] = (21-(ig-1)*(12/20))/60
        cmap[ig+ib+1, 1] = 1.0
        cmap[ig+ib+1, 2] = 0.5+(ig-1)*(0.3/20)

    for ir in range(21):
        cmap[ir+ig+ib+1, 0] = (8-(ir-1)*(7/20))/60
        cmap[ir+ig+ib+1, 1] = 0.5 + (ir-1)*(0.5/20)
        cmap[ir+ig+ib+1, 2] = 1

    cmap = plt.cm.hsv(cmap)
    return cmap
