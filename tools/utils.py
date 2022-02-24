import numpy as np
from numpy.linalg import norm
from scipy.io import wavfile
import ctypes
def SNR(x1, x2):
    """Compute Signal-to-noise ratio.
      :param x1: the audio signal.
      :param x2 : the noise signal
      :returns: SNR number.
      """
    return 20 * np.log10(norm(x1) / norm(x2))
