from typing import Sequence

import drjit as dr
import mitsuba as mi
from mitransient.utils import ArrayXf

from ..utils import indent


class PhasorImageBlock(mi.ImageBlock):
    """
    TODO(diego): docs
    """

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
    #     # self.put_(pos2, values, active)

    # def accum(self, value: mi.Float, index: mi.UInt32, active: mi.Bool):
    #     dr.scatter_reduce(dr.ReduceOp.Add, self.tensor().array,
    #                       value, index, active)

    # def put_(self, pos: mi.Point2f, values: Sequence[mi.Float], active: bool = True):
    #     # Check if all sample values are valid
    #     # if self.warn_negative or self.warn_invalid:
    #     #     is_valid = mi.Mask(True)

    #     #     if self.warn_negative:
    #     #         for k in range(self.channel_count()):
    #     #             is_valid &= values[k] >= -1e-5

    #     #     if self.warn_invalid:
    #     #         for k in range(self.channel_count()):
    #     #             is_valid &= dr.isfinite(values[k])

    #     #     if dr.any(active & ~is_valid):
    #     #         log_str = "Invalid sample value: ["
    #     #         for k in range(self.channel_count()):
    #     #             log_str += values[k]
    #     #             if k + 1 < self.channel_count():
    #     #                 log_str += ", "
    #     #         log_str += "]"
    #     #         mi.Log(mi.LogLevel.Warn, log_str)
    #     # if dr.any(~dr.isfinite(values)):
    #     #     mi.Log(mi.LogLevel.Warn, "Invalid sample value")

    #     # ====================================
    #     # Fast special case for the box filter
    #     # ====================================
    #     p = mi.Point2u(dr.floor(pos) - self.offset())

    #     index = dr.fma(p.y, self.size().x, p.x) * self.channel_count()

    #     active &= dr.all((0 <= p) & (p < self.size()))

    #     for k in range(self.channel_count()):
    #         dr.print('{value=}, {index=}, {aa=}',
    #                  value=values[k], index=index + k, aa=active, active=dr.any(~dr.isfinite(values)))
    #         self.accum(values[k], index + k, active)
