from typing import Sequence

import mitsuba as mi
from mitsuba import (
    Float,
    Int32,
    ScalarInt32,
    TensorXf,
)
import drjit as dr

from mitransient.render.phasor_image_block import PhasorImageBlock
from mitransient.utils import ArrayXf


class PhasorHDRFilm(mi.Film):
    r"""

    .. film-fourier_hdr_film:

    Fourier HDR Film

    TODO(diego): docs

    See also:
        - width/height (uint32)
        - crop_width/crop_height (uint32)
        - crop_offset_x/crop_offset_y (uint32)
        - sample_border (bool)
        - rfilters?
    """

    def __init__(self, props: mi.Properties):
        super().__init__(props)
        # FIXME generate from arbitrary list
        # maybe do Dist1D, or look at specfilm
        # or just dont do wl_mean+wl_sigma, allow for more arbitrary choices
        self.wl_mean = props.get("wl_mean", mi.Float(100.0))
        self.wl_sigma = props.get("wl_sigma", mi.Float(1000.0))
        self.temporal_bins = props.get("temporal_bins", mi.UInt32(4096))
        self.bin_width_opl = props.get("bin_width_opl", mi.Float(0.003))
        self.start_opl = props.get("start_opl", mi.Float(0))

        if not (self.crop_size().x == self.size().x and self.crop_size().y == self.size().y):
            mi.Log(mi.LogLevel.Error, "PhasorHDRFilm: crop_size must match size")
        if not (self.crop_offset().x == 0 and self.crop_offset().y == 0):
            mi.Log(mi.LogLevel.Error, "PhasorHDRFilm: crop_offset must be (0, 0)")
        if self.sample_border():
            mi.Log(mi.LogLevel.Error, "PhasorHDRFilm: sample_border must be False")

        import numpy as np
        nt = self.temporal_bins
        mean_idx = (nt * self.bin_width_opl) / self.wl_mean
        sigma_idx = (nt * self.bin_width_opl) / (self.wl_sigma * 6)
        # shift to center at zero, easier for low negative frequencies
        freq_min_idx = np.maximum(0, int(np.floor(mean_idx - 3 * sigma_idx)))
        freq_max_idx = np.minimum(
            nt // 2, int(np.ceil(mean_idx + 3 * sigma_idx)))

        frequencies = np.fft.fftfreq(nt, d=self.bin_width_opl)[
            freq_min_idx:freq_max_idx+1].astype(np.float32)

        mi.Log(mi.LogLevel.Info,
               f"PhasorHDRFilm: Using {len(frequencies)} wavelengths from {1/frequencies[-1]:.4f}m to {1/frequencies[0]:.4f}m")
        self.frequencies = ArrayXf([mi.Float(f) for f in frequencies])

    def prepare(self, aovs: Sequence[str]):
        if not mi.is_monochromatic:
            mi.Log(mi.LogLevel.Error,
                   "PhasorHDRFilm: Only monochromatic rendering supported")
        if len(aovs) != 0:
            mi.Log(mi.LogLevel.Error, "PhasorHDRFilm: AOVs not supported")
        alpha = mi.has_flag(self.flags(), mi.FilmFlags.Alpha)

        # Prepare steady film
        steady_hdrfilm_dict = {
            'type': 'hdrfilm',
            'width': self.size().x,
            'height': self.size().y,
            'pixel_format': "luminance" if mi.is_monochromatic else "rgb",
            'crop_offset_x': self.crop_offset().x,
            'crop_offset_y': self.crop_offset().y,
            'crop_width': self.crop_size().x,
            'crop_height': self.crop_size().y,
            'sample_border': self.sample_border(),
            'rfilter': self.rfilter()
        }
        self.steady: mi.Film = mi.load_dict(steady_hdrfilm_dict)
        self.steady.prepare(aovs)

        # Prepare phasor film
        base_channels = "L"
        extra_channels = "AW" if alpha else "W"

        channels = []
        for i in range(len(base_channels)):
            for j in range(len(self.frequencies)):
                channels.append(
                    base_channels[i] + f'_fq{j:03d}_Re')
                channels.append(
                    base_channels[i] + f'_fq{j:03d}_Im')

        for i in range(len(extra_channels)):
            channels.append(extra_channels[i])

        # NOTE(diego): aovs would go here
        # for i in range(len(aovs)):
        #     channels.append(aovs[i])

        self.phasors = PhasorImageBlock(
            size=self.size(),
            offset=self.crop_offset(),
            alpha=alpha,
            channel_count=len(channels),
            frequencies=self.frequencies,
        )
        self.channels = channels

        if len(set(channels)) != len(channels):
            mi.Log(mi.LogLevel.Error,
                   "Film::prepare_transient_(): duplicate channel name.")

        return len(self.channels)

    def clear(self):
        self.storage.clear()

    def develop(self, raw: bool = False):
        steady_image = self.steady.develop(raw=raw)
        phasors_image = self.develop_phasors_(raw=raw)

        return steady_image, phasors_image

    def develop_phasors_(self, raw: bool = False):
        if raw:
            return self.phasors.tensor()

        data = self.phasors.tensor()

        pixel_count = dr.prod(data.shape[0:-1])
        source_ch = data.shape[-1]
        # Remove alpha and weight channels
        alpha = mi.has_flag(self.flags(), mi.FilmFlags.Alpha)
        target_ch = source_ch - (ScalarInt32(2) if alpha else ScalarInt32(1))

        idx = dr.arange(Int32, pixel_count * target_ch)
        pixel_idx = idx // target_ch
        channel_idx = dr.fma(pixel_idx, -target_ch, idx)

        values_idx = dr.fma(pixel_idx, source_ch, channel_idx)
        weight_idx = dr.fma(pixel_idx, source_ch, source_ch - 1)

        weight = dr.gather(Float, data.array, weight_idx)
        values_ = dr.gather(Float, data.array, values_idx)

        values = values_ / dr.select((weight == 0.0), 1.0, weight)

        return TensorXf(values, tuple(list(data.shape[0:-1]) + [target_ch]))

    def add_transient_data(self, pos: mi.Vector2f, distance: mi.Float,
                           wavelengths: mi.UnpolarizedSpectrum, spec: mi.Spectrum,
                           ray_weight: mi.Float, active: mi.Bool):
        """
        Add a path's contribution to the film:
        * pos: pixel position
        * distance: distance traveled by the path (opl)
        * wavelengths: for spectral rendering, wavelengths sampled
        * spec: Spectrum / contribution of the path
        * ray_weight: weight of the ray given by the sensor
        * active: mask
        """
        pos_distance = (distance - self.start_opl)
        coords = mi.Vector3f(pos.x, pos.y, pos_distance)
        self.phasors.put(
            pos=coords,
            wavelengths=wavelengths,
            value=spec * ray_weight,
            alpha=mi.Float(0.0),
            # value should have the sample scale already multiplied
            weight=mi.Float(0.0),
            active=active,
        )

    def to_string(self):
        string = "PhasorHDRFilm[\n"
        string += f"  size = {self.size()},\n"
        string += f"  frequencies = {self.frequencies},\n"
        string += f"  start_opl = {self.start_opl},\n"
        string += f"]"
        return string

    def traverse(self, callback):
        super().traverse(callback)
        callback.put_parameter(
            "frequencies", self.frequencies, mi.ParamFlags.NonDifferentiable)
        callback.put_parameter(
            "start_opl", self.start_opl, mi.ParamFlags.NonDifferentiable)

    def parameters_changed(self, keys):
        super().parameters_changed(keys)


mi.register_film("phasor_hdr_film", lambda props: PhasorHDRFilm(props))
