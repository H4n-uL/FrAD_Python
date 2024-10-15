import numpy as np

def hanning_in_overlap(olap_len: int) -> np.ndarray: return 0.5 * (1 - np.cos(np.pi * np.arange(1, olap_len + 1) / (olap_len + 1)))
