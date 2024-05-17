import drjit as dr
import numpy as np

from mitsuba import Log, LogLevel, Object
from mitsuba import (
    Float,
    UInt32,
    ScalarUInt32,
    TensorXf,
    Int32,
    Mask,
    ReconstructionFilter,
)
from mitsuba import is_monochromatic, is_spectral
from mitsuba.math import RayEpsilon, linear_to_srgb  # type: ignore

from ..utils import indent, ArrayXu, ArrayXf


class TransientBlock(Object):
    """
    Transient Mitsuba 3's equivalent for ImageBlock class

    See Mitsuba 3's ImageBlock class for more information.
    - https://github.com/diegoroyo/mitsuba3/blob/master/include/mitsuba/render/imageblock.h
    See `transient_hdr_film` plugin for more information.
    """

    def __init__(
        self,
        size: np.ndarray,
        channel_count: ScalarUInt32,
        rfilter: ReconstructionFilter = None,
        compensate: bool = False,
        warn_negative: bool = False,
        warn_invalid: bool = False,
        border: bool = False,
        normalize: bool = False,
    ):
        super().__init__()
        self.m_offset = 0
        self.m_size = 0
        self.m_channel_count = channel_count
        self.m_compensate = compensate

        if warn_negative:
            Log(
                LogLevel.Warning,
                'Unused "warn_negative" because this class is mainly used in JIT variants',
            )

        if warn_invalid:
            Log(
                LogLevel.Warning,
                'Unused "warn_invalid" because this class is mainly used in JIT variants',
            )

        self.m_normalize = normalize
        self.m_rfilter = None
        self.m_data = None

        if border:
            Log(
                LogLevel.Error,
                f"Adding a border is not supported due to this class should not be used in scalar variants.",
            )

        self.set_size(size)
        self.configure_rfilter(rfilter, border)
        self.clear()

    def set_size(self, size):
        if np.all(self.m_size == size):
            return
        self.m_size = size

    # Reinitialize internal Tensor of image block's data
    def clear(self, force=False):
        del self.m_data

        size_data = self.m_size
        width = self.m_channel_count * np.prod(size_data)
        shape = tuple(list(size_data) + [self.m_channel_count])
        self.m_data = TensorXf(dr.zeros(Float, width), shape)

    # Configure rfilter's information
    def configure_rfilter(self, rfilter, border=False):
        if rfilter == None:
            raise (NameError("You need to define a reconstruction rfilter"))

        # Add check rfilter's type
        if not isinstance(rfilter, (list, tuple)):
            rfilter = [rfilter] * len(self.m_size)

        if len(self.m_size) != len(rfilter):
            raise (
                NameError(
                    "rfilter list should be equal to dimension or use only one rfilter for all dimensions"
                )
            )

        # Store the reconstruction filters
        self.m_rfilter = rfilter

    def data(self):
        return self.m_data

    def put(self, pos, wavelengths, value, alpha, weight, active):
        from mitsuba import unpolarized_spectrum

        spec_u = unpolarized_spectrum(value)

        if is_spectral:
            from mitsuba import spectrum_to_srgb

            rgb = spectrum_to_srgb(spec_u, wavelengths, active)
            values = [rgb[0], rgb[1], rgb[2], alpha, weight]
        elif is_monochromatic:
            values = [spec_u, alpha, weight]
        else:
            values = [spec_u[0], spec_u[1], spec_u[2], alpha, weight]

        self.put_(pos, values, active)

    def accum(self, value, index, active):
        if self.m_compensate:
            pass
        else:
            dr.scatter_reduce(
                dr.ReduceOp.Add, self.m_data.array, value, index, active)

    def put_(self, pos, value, active):
        rfilter_radius = np.array([f.radius() for f in self.m_rfilter])

        if np.all(rfilter_radius < (0.5 + RayEpsilon)):
            """
            Fast path for box filter
            """
            p = dr.floor(pos)

            offset = UInt32(0)
            for j in range(len(self.m_size)):
                offset += UInt32(p[j]) * np.prod(self.m_size[j + 1:])
                active &= (p[j] >= 0) & (p[j] < self.m_size[j])

            offset *= self.m_channel_count
            for k in range(self.m_channel_count):
                self.accum(value[k], offset + k, active)
        else:
            """
            Standard path when at least one filter has pixel extent
            """
            n = np.ceil(rfilter_radius - 0.5).astype(np.uint32)
            count = 2 * n + 1

            pos_i = dr.floor(pos) - n.astype(np.int32)
            # NOTE: Fixes issue with box filters combined with other filters
            rel_f = pos_i + 0.5 - pos - RayEpsilon

            # Precompute weights
            filter_weights = [Float(0.0)] * np.sum(count)
            base_index = 0
            for j in range(len(self.m_rfilter)):
                for i in range(count[j]):
                    index = np.uint32(base_index + i)
                    filter_weights[index] = self.m_rfilter[j].eval(
                        rel_f[j], active)
                    rel_f[j] += 1.0
                base_index += count[j]

            shape_data = self.m_data.shape

            # Accumulate samples
            def step_dimension(weight_, index_, index_filter_, active_, dim):
                weight = Float(weight_)
                index = UInt32(index_)
                active = Mask(active_)
                index_filter = np.uint32(index_filter_)

                if dim == len(self.m_rfilter):
                    # Store values in the block
                    for k in range(self.m_channel_count):
                        current_index = index * self.m_channel_count + k
                        self.accum(value[k] * weight, current_index, active)
                else:
                    for i in range(count[dim]):
                        w_filter = filter_weights[index_filter + i]
                        new_pos = UInt32(pos_i[dim]) + i
                        dim_index = new_pos * np.prod(self.m_size[dim + 1:])
                        current_active = (new_pos >= 0) & (
                            new_pos < shape_data[dim])
                        step_dimension(
                            weight * w_filter,
                            index + dim_index,
                            index_filter + count[dim],
                            active & current_active,
                            dim + 1,
                        )

            step_dimension(1.0, 0, 0, active, 0)

    def develop(self, gamma=False, raw=False):
        res = self.m_data

        if raw:
            return self.m_data

        pixel_count = dr.prod(self.m_data.shape[0:-1])
        source_ch = res.shape[-1]
        # Remove alpha and weight channels
        target_ch = source_ch - 2

        idx = dr.arange(UInt32, pixel_count * target_ch)
        pixel_idx = idx // target_ch
        channel_idx = dr.fma(pixel_idx, -target_ch, idx)

        values_idx = dr.fma(pixel_idx, source_ch, channel_idx)
        weight_idx = dr.fma(pixel_idx, source_ch, source_ch - 1)

        weight = dr.gather(Float, res.array, weight_idx)
        values_ = dr.gather(Float, res.array, values_idx)

        values = values_ / dr.select(dr.eq(weight, 0.0), 1.0, weight)

        if gamma and not is_monochromatic:
            values = linear_to_srgb(values)

        return TensorXf(values, list(self.m_data.shape[0:-1]) + [target_ch])

    def to_string(self):
        string = f"{type(self).__name__}[ \n"
        string += f"  size = {self.m_size}, \n"
        string += f"  normalize = {self.m_normalize}, \n"
        string += f"  compensate = {self.m_compensate}, \n"
        string += f"  rfilters = [ \n"
        for filter in self.m_rfilter:
            string += f"    {indent(filter, amount=4)}, \n"
        string += f"  ], \n"
        string += f"]"
        return string

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()
