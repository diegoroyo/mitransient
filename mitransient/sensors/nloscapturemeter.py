from __future__ import annotations  # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

from mitsuba import Log, LogLevel, is_spectral, Transform4f, ScalarVector2f
from drjit.scalar import Array3f as ScalarArray3f  # type: ignore

from typing import Tuple

from .nlossensor import NLOSSensor
from ..utils import indent


class NLOSCaptureMeter(NLOSSensor):
    r"""
    .. _sensor-nlos_capture_meter:

    NLOS Capture Meter (:monosp:`nlos_capture_meter`)
    -------------------------------------------------

    Attaches to a geometry (sensor should be child of the geometry). It measures uniformly-spaced points on such geometry.
    It is recommended to use a `rectangle` shape because the UV coordinates work better.

    The `nlos_capture_meter` should have a `film` children, which acts as the storage for the transient image. It is recommended to use `transient_hdr_film`.

    .. code-block:: python

        <shape type="rectangle">
            <sensor type="nlos_capture_meter">
                <film type="transient_hdr_film">
                    ...
                </film>
            </sensor>
        </shape>

    .. pluginparameters::

     * - account_first_and_last_bounces
       - |bool|
       - if True, the first and last bounces are accounted in the computation of the optical path length of the temporal dimension. This makes sense if you think of a NLOS setup. If False, the first and last bounces are not accounted (useful!)

     * - sensor_origin
       - |point|
       - position of the sensor (NLOS setup) in the world coordinate system

     * - original_film_{width|height}
       - |int|
       - special for confocal captures, you can ignore if you use one 
         illumination point or an exhaustive pattern. If you want to 
         simulate a confocal NLOS setup with NxM, you should use a 1x1 
         film instead of NxM, and point the laser to the point that you 
         want to capture. Then you should repeat the capture NxM times. We
         strongly recommend using TAL (see https://github.com/diegoroyo/tal), 
         and set `scan_type: confocal` in the tal render YAML configuration 
         file, which will handle all this automatically.

    See also the parameters for `transient_hdr_film`.
    """

    # TODO(diego): we assume the rays start in a vacuum
    # this is reasonable for NLOS scenes, but this can be changed
    # in the future if we want to support other media
    IOR_BASE = 1

    def __init__(self, props: mi.Properties):
        super().__init__(props)

        self.needs_sample_3: bool = False

        self.account_first_and_last_bounces: bool = \
            props.get('account_first_and_last_bounces', True)

        self.world_transform: Transform4f = \
            Transform4f().translate(
                props.get('sensor_origin', ScalarArray3f(0)))

        # Distance between the laser origin and the focusing point
        # Should be provided by the user if needed
        # see mitransient.nlos.focus_emitter_at_relay_wall
        self.laser_bounce_opl = mi.Float(0)
        self.laser_target = mi.Point3f(0)

        # Get the film size. Depends on if the capture is confocal or not
        self.original_film_width = props.get('original_film_width', None)
        self.original_film_height = props.get('original_film_height', None)
        if self.original_film_width is None or self.original_film_height is None:
            self.film_size: ScalarVector2f = ScalarVector2f(self.film().size())
            self.is_confocal = False
        else:
            self.film_size: ScalarVector2f = ScalarVector2f(
                self.original_film_width, self.original_film_height)
            self.is_confocal = True
            if self.film().size().x != 1 or self.film().size().y != 1:
                Log(LogLevel.Error,
                    f"Confocal configuration requires a film with size [1,1] instead of {self.film().size()}")

        dr.make_opaque(self.laser_bounce_opl,
                       self.laser_target, self.film_size)

    def _sensor_origin(self) -> ScalarArray3f:
        return self.world_transform.translation()

    def _pixel_to_sample(self, pixel: mi.Point2f) -> mi.Point2f:
        return pixel / self.film_size

    def _sample_direction(self,
                          time: mi.Float,
                          sample: mi.Point2f,
                          active: mi.Mask) -> Tuple[mi.Float, mi.Vector3f]:
        origin = self._sensor_origin()

        if self.is_confocal:
            # Confocal sample always center of the pixel
            target = self.laser_target
        else:
            # instead of continuous samples over the whole shape,
            # discretize samples so they only land on the center of the film's
            # "pixels"
            grid_sample = self._pixel_to_sample(
                dr.floor(sample * self.film_size) + 0.5)
            target = self.get_shape().sample_position(
                time, grid_sample, active
            ).p  # sampled position of PositionSample3f

        direction = target - origin
        distance = dr.norm(direction)
        direction /= distance
        return distance, direction

    def sample_ray_differential(
            self, time: mi.Float,
            sample1: mi.Float, sample2: mi.Point2f, sample3: mi.Point2f,
            active: mi.Bool = True) -> Tuple[mi.RayDifferential3f, mi.Color3f]:

        origin = self._sensor_origin()
        sensor_distance, direction = self._sample_direction(
            time, sample2, active)

        if is_spectral:
            wav_sample = mi.sample_shifted(sample1)
            wavelengths, wav_weight = mi.sample_rgb_spectrum(wav_sample)
        else:
            wavelengths = []
            wav_weight = 1.0

        if not self.account_first_and_last_bounces:
            time -= self.laser_bounce_opl + sensor_distance * self.IOR_BASE

        # NOTE: removed * dr.pi because there is no need to account for this
        return (
            mi.RayDifferential3f(origin, direction, time, wavelengths),
            mi.unpolarized_spectrum(wav_weight)  # * dr.pi
        )

    def pdf_direction(self,
                      it: mi.Interaction3f,
                      ds: mi.DirectionSample3f,
                      active: mi.Bool = True) -> mi.Float:
        # NOTE(diego): this could be used in sample_ray_differential
        # but other sensors do not do it (e.g. for a thin lens camera,
        # vignetting is not accounted for using this function)
        return self.shape().pdf_direction(it, ds, active)

    def eval(self, si: mi.SurfaceInteraction3f, active: mi.Bool = True) -> mi.Spectrum:
        return dr.pi / self.shape().surface_area()

    def bbox(self) -> mi.BoundingBox3f:
        return self.shape().bbox()

    def traverse(self, callback: mi.TraversalCallback):
        super().traverse(callback)
        callback.put_parameter(
            "needs_sample_3", self.needs_sample_3, mi.ParamFlags.NonDifferentiable)
        callback.put_parameter(
            "account_first_and_last_bounces",
            self.account_first_and_last_bounces, mi.ParamFlags.NonDifferentiable)
        callback.put_parameter(
            "is_confocal", self.is_confocal, mi.ParamFlags.NonDifferentiable)
        callback.put_parameter(
            "laser_bounce_opl", self.laser_bounce_opl, mi.ParamFlags.NonDifferentiable)
        callback.put_parameter(
            "laser_target", self.laser_target, mi.ParamFlags.NonDifferentiable)

    # NOTE(diego): fails on cuda_mono_double versions
    # def parameters_changed(self, keys):
    #     super().parameters_changed(keys)

    def to_string(self):
        string = f"{type(self).__name__}[\n"
        string += f"  laser_bounce_opl = {self.laser_bounce_opl}, \n"
        string += f"  account_first_and_last_bounces = {self.account_first_and_last_bounces}, \n"
        string += f"  is_confocal = {self.is_confocal}, \n"
        if self.is_confocal:
            string += f"  laser_target = {self.laser_target}, \n"
        string += f"  film = { indent(self.film()) }, \n"
        string += f"]"
        return string


mi.register_sensor('nlos_capture_meter', lambda props: NLOSCaptureMeter(props))
