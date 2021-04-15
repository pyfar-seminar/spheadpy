# %%
# Write exponential sweep to csv for testing.
# The sweep was manually inspected.
# The time signal was inspected for smootheness and maximum amplitudes of +/-1.
# The spectrum was inspected for the ripple at the edges of the frequency range
# (typical for time domain sweep generation) and the 1/f slope.
import numpy as np

import numpy.testing as npt
import pytest
import os
from pyfar import Signal

from pyfar.coordinates import Coordinates
from pyfar.spatial import samplings

from spheadpy import AKsphericalHead

def test_sph_great_circle_with_defaults():
    """Test sph_great_circle with default parameters."""

    sg = samplings.sph_great_circle(elevation=np.linspace(-90, 90, 19), gcd=10, radius=1,
                     azimuth_res=1, match=360)

    assert isinstance(sg, Coordinates)

def test_AKsphericalHead_with_defaults():
    """Test exponential sweep against manually verified reference."""
    sg = samplings.sph_great_circle(elevation=np.linspace(-90, 90, 19), gcd=10, radius=1,
                                    azimuth_res=1, match=360)

    shtf, offCenterParameter = AKsphericalHead(sg, ear = [85, -13], offCenter = False, r_0 = None, 
                                                    a = 0.0875, Nsh = 100, Nsamples = 1024, fs = 44100, c = 343)

    assert offCenterParameter == "False"

    assert shtf.cshape == (1024, )
    assert shtf.sampling_rate == 4410
    assert shtf.fft_norm == "none"
    