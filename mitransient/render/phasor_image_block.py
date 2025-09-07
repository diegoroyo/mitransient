from typing import Sequence

import drjit as dr
import mitsuba as mi
from mitransient.utils import ArrayXf

from ..utils import indent


class PhasorImageBlock(mi.ImageBlock):
    # This extends mi.ImageBlock to store phasor data
    # It uses two image channels per frequency: the first one for the real part,
    # the second one for the imaginary part

    def __init__(
        self,
        size: mi.ScalarVector3u,
        offset: mi.ScalarPoint3i,
        channel_count: int,
        frequencies=[],
        alpha: bool = False,
        warn_negative: bool = False,
        warn_invalid: bool = False
    ):
        self.frequencies = ArrayXf(frequencies)
        self.alpha = alpha
        # alpha = mi.has_flag(self.flags(), mi.FilmFlags.Alpha)
        # assert channel_count == 2 * len(frequencies) + (2 if alpha else 1), \
        #     "PhasorImageBlock: channel_count must match frequencies and alpha"
        super().__init__(
            size=size,
            offset=offset,
            channel_count=channel_count,
            rfilter=None,
            border=False,
            normalize=False,
            coalesce=False,
            compensate=False,
            warn_negative=warn_negative,
            warn_invalid=warn_invalid)

    def put(self, pos: mi.Point3f, wavelengths: mi.UnpolarizedSpectrum,
            value: mi.Spectrum, alpha: mi.Float,
            weight: mi.Float, active: bool = True):
        spec = mi.unpolarized_spectrum(value)
        spec = spec.x
        pos2 = mi.Point2f(pos.x, pos.y)
        opl = pos.z
        active &= dr.isfinite(opl)

        def fmod(x, y):
            return x - y * dr.floor(x / y)

        values = []
        for freq in self.frequencies:
            phase = fmod(-2 * dr.pi * freq * opl, 2 * dr.pi)
            values.append(mi.Float(spec * dr.cos(phase)))
            values.append(mi.Float(spec * dr.sin(phase)))

        if self.alpha:
            values += [alpha, weight]
        else:
            values += [weight]

        values = ArrayXf(values)

        super().put(pos2, values, active)
