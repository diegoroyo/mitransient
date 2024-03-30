import mitsuba as mi
import drjit as dr

from mitsuba import is_monochromatic, is_spectral
from mitransient.render.transient_block import TransientBlock


class TransientHDRFilm(mi.Film):
    """
    `transient_hdr_film` plugin
    ===========================

    Mitsuba 3 Transient's equivalent to Mitsuba 3's HDRFilm

    Stores two image blocks simultaneously:
    * self.steady: Accumulates all samples (sum over all the time dimension)
    * self.transient: Accumulates samples separating them in time bins (histogram)

    The `transient_hdr_film` plugin accepts the following parameters:
    * `temporal_bins` (integer): number of bins in the time dimension (histogram representation)
    * `bin_width_opl` (float): width of each bin in the time dimension (histogram representation)
    * `start_opl` (float): start of the time dimension (histogram representation)

    See also, from mi.Film:
    - https://github.com/diegoroyo/mitsuba3/blob/master/src/render/film.cpp
    - https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_films.html
    * `width` (integer)
    * `height` (integer)
    * `crop_width` (integer)
    * `crop_height` (integer)
    * `crop_offset_x` (integer)
    * `crop_offset_y` (integer)
    * `sample_border` (bool)
    * `rfilter` (rfilter)
    """

    def __init__(self, props):
        super().__init__(props)
        # NOTE: Also inherits properties from mi.Film (see documentation for this class above)
        self.temporal_bins = props.get("temporal_bins", mi.UInt32(2048))
        self.bin_width_opl = props.get("bin_width_opl", mi.Float(0.003))
        self.start_opl = props.get("start_opl", mi.UInt32(0))

        dr.make_opaque(self.temporal_bins, self.bin_width_opl, self.start_opl)

    def end_opl(self):
        return self.start_opl + self.bin_width_opl * self.temporal_bins

    def add_transient_data(self, spec, distance, wavelengths, active, pos, ray_weight):
        """
        Add a path's contribution to the film
        * spec: Spectrum / contribution of the path
        * extra_weight: WIP. Hidden Geometry Rejection Sampling stuff.
        * distance: distance traveled by the path (opl)
        * wavelengths: for spectral rendering, wavelengths sampled
        * active: mask
        * pos: pixel position
        * ray_weight: weight of the ray given by the sensor
        """
        idd = (distance - self.start_opl) / self.bin_width_opl
        coords = mi.Vector3f(pos.x, pos.y, idd)
        mask = (idd >= 0) & (idd < self.temporal_bins)
        self.transient.put(
            pos=coords,
            wavelengths=wavelengths,
            value=spec * ray_weight,
            alpha=mi.Float(0.0),
            # value should have the sample scale already multiplied
            weight=mi.Float(0.0),
            active=active & mask,
        )

    def prepare(self, aovs):
        """Called before the rendering starts (stuff related to steady-state rendering)"""
        # NOTE could be done with mi.load_dict where type='hdrfilm' and the rest of the properties
        props = mi.Properties("hdrfilm")
        props["width"] = self.size().x
        props["height"] = self.size().y
        props["crop_width"] = self.crop_size().x
        props["crop_height"] = self.crop_size().y
        props["crop_offset_x"] = self.crop_offset().x
        props["crop_offset_y"] = self.crop_offset().y
        props["sample_border"] = self.sample_border()
        props["pixel_format"] = "luminance" if is_monochromatic else "rgb"
        props["rfilter"] = self.rfilter()
        self.steady = mi.PluginManager.instance().create_object(props)
        self.steady.prepare(aovs)

    def prepare_transient(self, size, rfilter):
        """
        Called before the rendering starts (stuff related to transient rendering)
        This function also allocates the needed number of channels depending on the variant
        """
        channel_count = 3 if is_monochromatic else 5
        self.transient = TransientBlock(
            size=size, channel_count=channel_count, rfilter=rfilter
        )

    def traverse(self, callback):
        # TODO: all the parameters are set as NonDifferentiable by default
        super().traverse(callback)
        callback.put_parameter(
            "temporal_bins", self.temporal_bins, mi.ParamFlags.NonDifferentiable
        )
        callback.put_parameter(
            "bin_width_opl", self.bin_width_opl, mi.ParamFlags.NonDifferentiable
        )
        callback.put_parameter(
            "start_opl", self.start_opl, mi.ParamFlags.NonDifferentiable
        )

    def parameters_changed(self, keys):
        super().parameters_changed(keys)

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


mi.register_film("transient_hdr_film", lambda props: TransientHDRFilm(props))
