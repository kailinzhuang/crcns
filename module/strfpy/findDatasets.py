import os
import re

def find_datasets(rootdir, stimDir, expr=None):
    
    if expr is None:
        expr = ''
    
    datasets = []
    dcnt = 1
    fnames = os.listdir(rootdir)
    
    for fname in fnames:
        if not fname.startswith('.') and not fname.startswith('..'):
            if os.path.isdir(os.path.join(rootdir, fname)):
                
                subdir = os.path.join(rootdir, fname)
                srPairs = get_sr_files(subdir, stimDir)
                
                if isinstance(srPairs, dict):
                    
                    matched = True
                    if expr:
                        mstr = re.search(expr, subdir)
                        matched = bool(mstr)
                        
                    if matched:
                        ds = {'dirname': subdir, 'srPairs': srPairs}
                        datasets.append(ds)
                        dcnt += 1
                else:
                    ds = find_datasets(subdir, stimDir, expr)
                    for j in range(len(ds)):
                        datasets.append(ds[j])
                        dcnt += 1
    
    return datasets

def get_sr_files(dataDir, stimDir):
    
    srPairs = -1
    stimLinkFiles = get_filenames(dataDir, 'stim[0-9]*', 1)
    respFiles = get_filenames(dataDir, 'spike[0-9]*', 1)
    
    if isinstance(stimLinkFiles, list) and isinstance(respFiles, list) and (len(stimLinkFiles) == len(respFiles)):
        
        srPairs = {}
        nFiles = len(stimLinkFiles)
        stimFiles = [None] * nFiles
        
        for k in range(nFiles):
            
            # read stim file and get path to .wav file
            stimLinkFile = stimLinkFiles[k]
            with open(stimLinkFile, 'r') as f:
                wavFile = f.readline().strip()
            stimFiles[k] = os.path.join(stimDir, wavFile)
            
        srPairs['stimFiles'] = stimFiles
        srPairs['respFiles'] = respFiles
        
    return srPairs



def get_filenames(datadir, regex, prependDataDir=False):
    dlist = os.listdir(datadir)
    fileCount = 0
    fileNames = []

    for fname in dlist:
        if not fname.startswith("."):
            mstr = re.search(regex, fname)
            if mstr:
                fileCount += 1
                if prependDataDir:
                    fname = os.path.join(datadir, fname)
                fileNames.append(fname)

    if fileCount == 0:
        fileNames = -1

    return fileNames

