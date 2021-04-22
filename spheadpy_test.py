# %%
import numpy as np

import numpy.testing as npt
import pytest
import os
from pyfar import Signal

from pyfar import Coordinates
from pyfar import samplings

from HRTF import sphead

def test_sph_great_circle_with_defaults():
    """Test sph_great_circle with default parameters."""

    sg = samplings.sph_great_circle(elevation=np.linspace(-90, 90, 19), gcd=10, radius=1,
                     azimuth_res=1, match=360)

    assert isinstance(sg, Coordinates)

def test_sphead_with_defaults():
    """Test sphead with default paramters."""
    sg = samplings.sph_great_circle(elevation=np.linspace(-90, 90, 19), gcd=10, radius=1,
                                    azimuth_res=1, match=360)

    shtf, offCenterParameter = sphead(sg, ear = [85, -13], offCenter = False, r_0 = None,
                                                    a = 0.0875, Nsh = 100, Nsamples = 1024, fs = 44100, c = 343)

    assert offCenterParameter == False

    assert shtf.cshape == (440, 2)
    assert shtf.time.shape == (440, 2, 1024)
    assert shtf.n_samples == 1024
    assert shtf.sampling_rate == 44100
    assert shtf.fft_norm == "none"

test_sphead_with_defaults()