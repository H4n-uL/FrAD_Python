import numpy as np
import math

def hanning_in_overlap(olap_len): return np.array([0.5 * (1.0 - math.cos(math.pi * i / (olap_len + 1))) for i in range(1, olap_len + 1)])