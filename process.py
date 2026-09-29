import numpy as np
from scipy.signal import find_peaks


class Process:
    def __init__(self):
        pass
    # DC offset removal
    def offset_removal(self, data):
        offset_dc_removal = data- np.nanmean(data)
        return offset_dc_removal
    