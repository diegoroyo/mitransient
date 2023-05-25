import drjit as dr
import numpy as np

from mitsuba import Float, UInt32, TensorXf, Int32, Mask, ReconstructionFilter
from mitsuba import is_monochromatic, is_spectral
from mitsuba.math import RayEpsilon, linear_to_srgb

from mitransient.utils import ArrayXu, ArrayXf


class TransientBlock:
    def __init__(self,
                 size: np.ndarray,
                 channel_count: UInt32,
                 channel_use_weights: bool = True,
                 rfilter: ReconstructionFilter = None,
                 warn_negative: bool = False,
                 warn_invalid: bool = False,
                 border: bool = True,
                 normalize: bool = False):
        self.m_offset = 0
        self.m_size = 0
        self.m_channel_count = channel_count
        self.m_channel_use_weights = channel_use_weights
        self.m_warn_negative = warn_negative
        self.m_warn_invalid = warn_invalid
        self.m_normalize = normalize
        self.m_rfilter = None
        self.m_data = None

        self.set_size(size)
        self.configure_rfilter(rfilter, border)
        self.clear()

    def set_size(self, size):
        if np.all(self.m_size == size):
            return
        self.m_size = size
        self.m_offset = np.array([0]*len(self.m_size), dtype=np.uint32)

    # Reinitialize internal Tensor of image block's data
    def clear(self, force=False):
        # if type(self.m_data) != type(None):
        del self.m_data

        size_data = self.m_size + 2 * \
            (self.m_border_size if force else self.m_original_border_size)
        width = self.m_channel_count * np.prod(size_data)
        shape = tuple(list(size_data) + [self.m_channel_count])
        self.m_data = TensorXf(dr.zeros(Float, width), shape)

    # Configure rfilter's information
    def configure_rfilter(self, rfilter, border=True):
        if rfilter == None:
            raise (NameError('You need to define a reconstruction rfilter'))

        # Add check rfilter's type
        if not isinstance(rfilter, (list, tuple)):
            rfilter = [rfilter] * len(self.m_size)

        if len(self.m_size) != len(rfilter):
            raise (NameError(
                'rfilter list should be equal to dimension or use only one rfilter for all dimensions'))

        # self.m_border_size = ArrayXu([f.border_size() for f in rfilter]) if rfilter != None and border else ArrayXu(0)
        border_size = np.array([f.border_size() for f in rfilter]
                               ) if rfilter != None and border else np.array(0)

        if self.m_rfilter == None:
            # Save bigger border for developing later
            self.m_border_size = border_size
            self.m_original_border_size = border_size
            self.m_border_offset = np.array(
                [0]*len(self.m_size), dtype=np.uint32)
        else:
            # Update offset to account for a smaller rfilter's radius
            if dr.any(border_size != self.m_original_border_size):
                self.m_border_offset = np.array(
                    self.m_original_border_size - border_size, dtype=np.uint32)
            self.m_border_size = border_size

        self.m_rfilter = rfilter
        # Prepare tensor
        self.rfilter_radius = np.array([f.radius() for f in self.m_rfilter])
        # rfilter_size = np.ceil(2 * self.rfilter_radius).astype(np.uint32) + 1
        rfilter_size = np.ceil(
            (self.rfilter_radius - 2.0 * RayEpsilon) * 2.0).astype(np.uint32)

        width = np.sum(rfilter_size, dtype=np.uint32)
        self.m_weights = [Float(0.0)] * width

    def data(self):
        return self.m_data

    def put(self, pos, wavelengths, value, alpha, active):
        from mitsuba import unpolarized_spectrum
        spec_u = unpolarized_spectrum(value)

        if is_spectral:
            from mitsuba import spectrum_to_srgb
            rgb = spectrum_to_srgb(spec_u, wavelengths, active)
        elif is_monochromatic:
            rgb = spec_u.x
        else:
            rgb = spec_u

        values = [rgb[0], rgb[1], rgb[2], alpha, 1.0]
        return self.put_(pos, values, active)

    def put_(self, pos_, value, active):
        border_size = ArrayXf(self.m_border_size.tolist())
        border_offset = ArrayXf(self.m_border_offset.tolist())
        offset = ArrayXf(self.m_offset.tolist())
        rfilter_radius = ArrayXf(self.rfilter_radius.tolist())
        # size = ArrayXu(self.m_size.tolist()) + 2 * border_size
        size = ArrayXu(self.m_size.tolist()) + 2 * \
            ArrayXu(self.m_original_border_size.tolist())
        # Convert to pixel coordinates within the image block
        pos = pos_ - (offset - (border_size + border_offset) + 0.5)
        # pos = pos_ - (self.m_offset - border_size)

        # if dr.any(rfilter_radius > (0.5 + RayEpsilon))[0]:
        if np.any(self.rfilter_radius > (0.5 + RayEpsilon)):
            # Determine the affected range of pixels
            lo = dr.maximum(dr.ceil(pos - rfilter_radius), 0)
            hi = dr.minimum(dr.floor(pos + rfilter_radius),
                            size - (1 + border_offset))
            lo = ArrayXu(lo)
            hi = ArrayXu(hi)

            # n = dr.ceil((rfilter_radius - 2.0 * RayEpsilon) * 2.0)
            n = np.ceil((self.rfilter_radius - 2.0 * RayEpsilon)
                        * 2.0).astype(np.uint32)

            # Precompute rfilter weights
            base = lo - pos
            base_index = 0
            for j in range(len(self.m_rfilter)):
                # for i in range(Int32(n[j])[0]):
                for i in range(n[j]):
                    p = UInt32(base[j]) + i
                    index = np.uint32(base_index + i)
                    self.m_weights[index] = self.m_rfilter[j].eval(p, active)
                base_index += n[j]

            # Normalize rfilter weights if requested (per dimension)
            # Need to be tested !!!
            if self.m_normalize:
                factor = Float(1.0)
                base_index = UInt32(0)
                for j in range(len(self.m_rfilter)):
                    index = UInt32(n[j])[0]
                    factor *= dr.hsum(self.m_weights[base_index[0]
                                      :base_index[0]+index])
                    base_index += UInt32(n[j])

                factor = dr.rcp(factor)

                for i in range(Int32(n[0])[0]):
                    self.m_weights[i] *= factor[0]

            idxs = dr.zeros(UInt32, len(self.m_rfilter))
            while True:
                # Gather weigths
                weigth = Float(1.0)
                base_index = 0
                for j in range(len(self.m_rfilter)):
                    weigth *= self.m_weights[base_index+idxs[j]]
                    base_index += n[j]

                # Gather offset of values
                offset = UInt32(0)
                enabled = Mask(active)
                for j in range(len(self.m_rfilter)-1, -1, -1):
                    offset += UInt32((idxs[j] + lo[j])
                                     * UInt32(dr.prod(size[j+1:])))
                    enabled &= (idxs[j] + lo[j]) <= hi[j]
                offset *= UInt32(self.m_channel_count)

                # Scatter values in imageblock
                for k in range(self.m_channel_count):
                    dr.scatter_reduce(dr.ReduceOp.Add, self.m_data.array, value[k] * weigth, offset + UInt32(k),
                                      enabled)

                # Update
                j = 0
                for j in range(len(n)):
                    idxs[j] += 1
                    if idxs[j] < Int32(n[j])[0]:
                        break
                    idxs[j] = 0
                    j += 1
                if j == len(n):
                    break
        else:
            lo = dr.ceil(pos - 0.5)
            offset = UInt32(0)

            for j in range(len(self.m_size)-1, -1, -1):
                offset += UInt32(lo[j]) * UInt32(dr.prod(size[j+1:]))

            offset *= UInt32(self.m_channel_count)
            enabled = active & dr.all((lo >= 0) & (lo < size))
            for k in range(self.m_channel_count):
                dr.scatter_reduce(dr.ReduceOp.Add, self.m_data.array, value[k], offset + UInt32(k),
                                  enabled)
        return active

    def develop(self, gamma=False, integer=False, raw=False):
        res = self.m_data
        dr.eval(res)

        if raw:
            return res

        pixel_count = dr.prod(res.shape[0:-1])
        ch = res.shape[-1]
        target_ch = ch - 2

        i = dr.arange(UInt32, pixel_count * target_ch)
        i_channel = i % target_ch
        weight_idx = (i // target_ch) * ch + 4
        values_idx = (i // target_ch) * ch + i_channel

        weight = dr.gather(Float, res.array, weight_idx)
        values = dr.gather(Float, res.array, values_idx)

        if self.m_channel_use_weights:
            values = (values / weight) & (weight > 0.0)

        if gamma:
            values = linear_to_srgb(values)

        res = TensorXf(values, list(res.shape[0:-1]) + [target_ch])
        crop_size = tuple([np.s_[:] if bi == 0 else np.s_[bi:-bi]
                          for bi in self.m_original_border_size])
        return res[crop_size]

    def __str__(self):
        # TODO update
        # return f'ImageBlockND[size = {self.m_size}]'
        return f'''ImageBlockND[
        size = {self.m_size}
        offset = {self.m_offset}
        borderSize = {self.m_border_size}
        originalbordersize = {self.m_original_border_size}
        rfilter = {self.m_rfilter}
        weigths = {self.m_weights}
        data = {self.m_data}
]
'''
