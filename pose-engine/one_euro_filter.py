"""
1-Euro Filter — Adaptive low-pass filter for real-time pose smoothing.
Reduces jitter WITHOUT adding lag during fast movement.
Perfect for clinical joint-angle measurement.

Reference: Casiez et al. "The 1€ Filter" (2012)
"""
import numpy as np


class OneEuroFilter:
    """
    Adaptive low-pass filter.
    mincutoff = minimum cutoff frequency (removes jitter at rest)
    beta = speed coefficient (higher = more responsive to fast motion)
    dcutoff = derivative cutoff (smooths velocity signal)
    """

    def __init__(self, freq: float = 30.0, mincutoff: float = 1.0,
                 beta: float = 0.05, dcutoff: float = 1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff

        self.x = None       # filtered signal
        self.dx = None      # filtered derivative
        self.last_time = None

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float, timestamp: float = None) -> float:
        if self.x is None:
            self.x = x
            self.dx = 0.0
            self.last_time = timestamp if timestamp is not None else 0.0
            return x

        dt = (timestamp if timestamp is not None else 0.0) - self.last_time
        if dt <= 0:
            dt = 1.0 / self.freq

        # Filtered derivative
        dx = (x - self.x) / dt
        edx = self.dx + self._alpha(self.dcutoff) * (dx - self.dx)

        # Adaptive cutoff based on movement speed
        cutoff = self.mincutoff + self.beta * abs(edx)
        self.x += self._alpha(cutoff) * (x - self.x)
        self.dx = edx
        self.last_time = timestamp if timestamp is not None else 0.0
        return self.x


class OneEuroFilter3D:
    """Independent 1-Euro filters for X, Y, Z coordinates."""

    def __init__(self, freq=30.0, mincutoff=1.0, beta=0.05, dcutoff=1.0):
        self.fx = OneEuroFilter(freq, mincutoff, beta, dcutoff)
        self.fy = OneEuroFilter(freq, mincutoff, beta, dcutoff)
        self.fz = OneEuroFilter(freq, mincutoff, beta, dcutoff)

    def filter(self, x: float, y: float, z: float,
               timestamp: float = None) -> tuple:
        return (
            self.fx.filter(x, timestamp),
            self.fy.filter(y, timestamp),
            self.fz.filter(z, timestamp)
        )
