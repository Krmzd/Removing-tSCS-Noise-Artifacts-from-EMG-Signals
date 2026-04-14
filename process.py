import numpy as np
from scipy import signal
from scipy.signal import find_peaks


class Process:
    def __init__(self):
        pass
    # DC offset removal
    def offset_removal(self, data):
        offset_dc_removal = data- np.nanmean(data)
        return offset_dc_removal
    
    def rectification(self, data):
        full_wave_rectify = np.abs(data)
        return full_wave_rectify
   
    
    # Segmenting
    def segment(self, data, stim_artifact, number_of_stims= 20, fs= 2000):
        segments = []
        for i in range(number_of_stims):
            segment = data[stim_artifact + int(-50 / 1000 * fs ) : stim_artifact + int(200 / 1000 * fs)]
            if len(segment) < 500:
                break
            segments.append(segment)
            stim_artifact = stim_artifact + 20000 
        return np.array(segments)
    
    # time of stimulation artifact
    def stimulation_artifact_detection(self, data):
        artifact_thresh = 1
        peaks, properties= find_peaks(data, height= artifact_thresh)
        for peak in peaks:
            if peak > 100:
                first_peak_index = int(peak)
                break
        return first_peak_index
        