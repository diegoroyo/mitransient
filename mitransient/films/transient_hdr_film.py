from typing import Sequence

import mitsuba as mi
import drjit as dr

from mitransient.render.transient_image_block import TransientImageBlockTwo


class TransientHDRFilm(mi.Film):

    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.temporal_bins = props.get("temporal_bins", mi.UInt32(2048))
        self.bin_width_opl = props.get("bin_width_opl", mi.Float(0.003))
        self.start_opl = props.get("start_opl", mi.UInt32(0))

    def end_opl(self):
        return self.start_opl + self.bin_width_opl * self.temporal_bins

    def base_channels_count(self):
        return self.steady.base_channels_count()

    def prepare(self, aovs: Sequence[str]):
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

        # Prepare transient image block
        channels = self.prepare_transient_(aovs)
        return channels

    def prepare_transient_(self, aovs: Sequence[str]):
        alpha = mi.has_flag(self.flags(), mi.FilmFlags.Alpha)
        base_channels = 5 if alpha else 4

        base_channel_names = "RGBAW" if alpha else "RGBW"

        channels = []
        for i in range(base_channels):
            channels.append(base_channel_names[i])
        
        for i in range(len(aovs)):
            channels.append(aovs[i])

        crop_offset_with_time_dimension = mi.ScalarPoint3i(self.crop_offset().x, self.crop_offset().y, 0)
        crop_size_with_time_dimension = mi.ScalarVector3u(self.size().x, self.size().y, self.temporal_bins)

        self.transient_storage = TransientImageBlockTwo(
            size_with_time_dimension=crop_size_with_time_dimension,
            offset=crop_offset_with_time_dimension,
            channel_count=len(channels), 
            rfilter=self.rfilter()
        )
        self.channels = channels

        if len(set(channels)) != len(channels):
            mi.Log(mi.LogLevel.Error, "Film::prepare_transient_(): duplicate channel name.")

        return len(self.channels)
    
    def clear(self):
        self.steady.clear()

        if self.transient_storage:
            self.transient_storage.clear()

    def develop(self, raw: bool = False):
        steady_image = self.steady.develop(raw)
        transient_image = self.develop_transient_(raw = True)

        return steady_image, transient_image
    
    def develop_transient_(self, raw: bool = False):
        if not self.transient_storage:
            mi.Log(mi.LogLevel.Error, "No transient storage allocated, was prepare_transient_() called first?")

        if raw:
            return self.transient_storage.tensor
        else:
            # TODO(JORGE): implement develop for transient image
            mi.Log(mi.LogLevel.Error, "TransientHDRFilm only allows to develop image buffer in raw format.")

    def add_transient_data(self, pos: mi.Vector2f, distance: mi.Float, wavelengths: mi.UnpolarizedSpectrum, spec: mi.Spectrum, ray_weight: mi.Float, active: mi.Bool):
        """
        Add a path's contribution to the film:
        * pos: pixel position
        * distance: distance traveled by the path (opl)
        * wavelengths: for spectral rendering, wavelengths sampled
        * spec: Spectrum / contribution of the path
        * ray_weight: weight of the ray given by the sensor
        * active: mask
        """
        pos_distance = (distance - self.start_opl) / self.bin_width_opl
        coords = mi.Vector3f(pos.x, pos.y, pos_distance)
        mask = (pos_distance >= 0) & (pos_distance < self.temporal_bins)
        self.transient_storage.put(
            pos=coords,
            wavelengths=wavelengths,
            value=spec * ray_weight,
            alpha=mi.Float(0.0),
            weight=mi.Float(0.0), # value should have the sample scale already multiplied
            active=active & mask,
        )

    def to_string(self):
        string = "TransientHDRFilm[\n"
        string += f"  size = {self.size()},\n"
        string += f"  crop_size = {self.crop_size()},\n"
        string += f"  crop_offset = {self.crop_offset()},\n"
        string += f"  sample_border = {self.sample_border()},\n"
        string += f"  filter = {self.rfilter()},\n"
        string += f"  temporal_bins = {self.temporal_bins},\n"
        string += f"  bin_width_opl = {self.bin_width_opl},\n"
        string += f"  start_opl = {self.start_opl},\n"
        string += f"]"
        return string

    def traverse(self, callback):
        super().traverse(callback)
        callback.put_parameter("temporal_bins", self.temporal_bins, mi.ParamFlags.NonDifferentiable)
        callback.put_parameter("bin_width_opl", self.bin_width_opl, mi.ParamFlags.NonDifferentiable)
        callback.put_parameter("start_opl", self.start_opl, mi.ParamFlags.NonDifferentiable)

    def parameters_changed(self, keys):
        super().parameters_changed(keys)

mi.register_film("transient_hdr_film", lambda props: TransientHDRFilm(props))