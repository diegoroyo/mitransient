from typing import Sequence

import drjit as dr
import mitsuba as mi

from ..utils import indent


class TransientImageBlock(mi.Object):
    """
    mitransient's equivalent for ImageBlock class

    See Mitsuba 3's ImageBlock class for more information.
    - https://github.com/diegoroyo/mitsuba3/blob/master/include/mitsuba/render/imageblock.h
    See ``transient_hdr_film`` plugin for more information.
    """

    def __init__(
        self,
        size_xyt: mi.ScalarVector3u,
        offset_xyt: mi.ScalarPoint3i,
        channel_count: int,
        rfilter: mi.ReconstructionFilter,
        border: bool = False,
        # normalize: bool = False,
        # coalesce: bool = False,
        # compensate: bool = False,
        warn_negative: bool = False,
        warn_invalid: bool = False
    ):
        super().__init__()
        self.offset_xyt = offset_xyt
        self.size_xyt = mi.ScalarVector3u(0)
        self.channel_count = channel_count
        self.rfilter = rfilter
        # TODO(diego): figure out if we need normalize/coalesce/compensate
        # for now, they don't do anything
        # self.normalize = normalize
        # self.coalesce = coalesce
        # self.compensate = compensate
        self.warn_negative = warn_negative
        self.warn_invalid = warn_invalid

        if rfilter and rfilter.is_box_filter():
            self.rfilter = None

        self.border_size = self.rfilter.border_size() if (self.rfilter and border) else 0

        self.set_size(size_xyt)
        self.clear()

    def clear(self):
        border_size_ScalarPoint3 = mi.ScalarVector3u(
            self.border_size, self.border_size, 0)
        size_ext = self.size_xyt + 2 * border_size_ScalarPoint3

        size_flat = self.channel_count * dr.prod(size_ext)
        shape = (size_ext.y, size_ext.x, size_ext.z, self.channel_count)

        self.tensor = mi.TensorXf(dr.zeros(mi.Float, size_flat), shape)
        # Compensation is not implented: https://github.com/mitsuba-renderer/mitsuba3/blob/b2ec619c7ba612edb1cf820463b32e5a334d8471/src/render/imageblock.cpp#L80

    def set_size(self, size_xyt: mi.ScalarVector3u):
        if dr.all(size_xyt == self.size_xyt):
            return

        self.size_xyt = size_xyt

    def accum(self, value: mi.Float, index: mi.UInt32, active: mi.Bool):
        dr.scatter_reduce(dr.ReduceOp.Add, self.tensor.array,
                          value, index, active)

    def put(self, pos: mi.Point3f, wavelengths: mi.UnpolarizedSpectrum,
            value: mi.Spectrum, alpha: mi.Float,
            weight: mi.Float, active: bool = True):
        spec_u = mi.unpolarized_spectrum(value)

        if mi.is_spectral:
            rgb = mi.spectrum_to_srgb(spec_u, wavelengths, active)
            values = [rgb.x, rgb.y, rgb.z, alpha, weight]
        elif mi.is_monochromatic:
            values = [spec_u.x, alpha, weight]
        else:
            values = [spec_u.x, spec_u.y, spec_u.z, alpha, weight]

        self.put_(pos, values, active)

    def put_(self, pos: mi.Point3f, values: Sequence[mi.Float], active: bool = True):
        # Check if all sample values are valid
        if self.warn_negative or self.warn_invalid:
            is_valid = True

            if self.warn_negative:
                for k in range(self.channel_count):
                    is_valid &= values[k] >= -1e-5

            if self.warn_invalid:
                for k in range(self.channel_count):
                    is_valid &= dr.isfinite(values[k])

            if dr.any(active and not is_valid):
                log_str = "Invalid sample value: ["
                for k in range(self.channel_count):
                    log_str += values[k]
                    if k + 1 < self.channel_count:
                        log_str += ", "
                log_str += "]"
                mi.Log(mi.LogLevel.Warn, log_str)

        # ====================================
        # Fast special case for the box filter
        # ====================================
        if not self.rfilter:
            p = mi.Point3u(dr.floor(pos) - self.offset_xyt)

            index = dr.fma(p.y, self.size_xyt.x, p.x)
            index = dr.fma(index, self.size_xyt.z, p.z) * self.channel_count

            active &= dr.all((0 <= p) & (p < self.size_xyt))

            for k in range(self.channel_count):
                self.accum(values[k], index + k, active)
        else:
            mi.Log(mi.LogLevel.Error, "TransientImageBlock::put_(): using a rfilter but it is not supported. If you need this, please open an issue on GitHub.")

    def to_string(self):
        string = f"{type(self).__name__}[\n"
        string += f"  offset_xyt = {self.offset_xyt}"
        string += f"  size_xyt = {self.size_xyt}, \n"
        string += f"  channel_count = {self.channel_count}, \n"
        string += f"  border_size = {self.border_size}, \n"
        string += f"  normalize = {self.normalize}, \n"
        string += f"  coalesce = {self.coalesce}, \n"
        string += f"  compensate = {self.compensate}, \n"
        string += f"  warn_negative = {self.warn_negative}, \n"
        string += f"  warn_invalid = {self.warn_invalid}, \n"
        if self.rfilter:
            string += f"  rfilter = {indent(self.rfilter, amount=4)} \n"
        else:
            string += f"  rfilter = BoxFilter[] \n"
        string += "]"
        return string

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()
