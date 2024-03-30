from __future__ import annotations  # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

from mitsuba import Log, LogLevel, is_spectral, Transform4f, ScalarVector2u
from drjit.scalar import Array3f as ScalarArray3f  # type: ignore

from typing import Tuple

from .nlossensor import NLOSSensor
from ..utils import indent


class NLOSCaptureMeter(NLOSSensor):
    """
        `nlos_capture_meter` plugin
        ===========================

        Attaches to a geometry (sensor should be child of the geometry).
        Measures uniformly-spaced points on such geometry.
        It is recommended to use a `rectangle` shape, the UV coordinates work better.

        The `nlos_capture_meter` should have a `film` children, which acts as the
        storage for the transient image. It is recommended to use `transient_hdr_film`.

        <shape type="rectangle">
            <sensor type="nlos_capture_meter">
                <film type="transient_hdr_film">
                    ...
                </film>
            </sensor>
        </shape>

        The `nlos_capture_meter` plugin accepts the following parameters:
        * `account_first_and_last_bounces` (boolean): if `True`, the first and last bounces are accounted
            in the computations of the optical path length of the temporal dimension.
            This makes sense if you think of a NLOS setup.
            If `False`, the first and last bounces are not accounted (useful!)
        * `confocal` (boolean): if `True`, the sensor only measures the point where the laser is pointed to.
            To model multiple illumination points, repeat the `nlos_capture_meter` sensor or
            render multiple times (see https://github.com/diegoroyo/tal), search for `scan_type`.
        * `sensor_origin` (point): position of the sensor (NLOS setup) in the world coordinate system

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
            Transform4f.translate(
                props.get('sensor_origin', ScalarArray3f(0)))

        # Distance between the laser origin and the focusing point
        # Should be provided by the user if needed
        # see mitransient.nlos.focus_emitter_at_relay_wall
        self.laser_bounce_opl = mi.Float(0)

        # Confocal setup pixel if desired by the user (default -1, no confocal)
        self.confocal_pixel = props.get(
            'confocal_pixel', mi.ScalarPoint2f(-1, -1))
        self.is_confocal: bool = self.confocal_pixel.x >= 0 and self.confocal_pixel.y >= 0

        film_size: ScalarVector2u = self.film().size()
        if self.is_confocal and film_size.x != 1 and film_size.y != 1:
            Log(LogLevel.Error,
                f"Confocal configuration requires a film with size [1,1] instead of {film_size}")

        dr.make_opaque(self.laser_bounce_opl, self.confocal_pixel)

    def _sensor_origin(self) -> ScalarArray3f:
        return self.world_transform.translation()

    def _pixel_to_sample(self, pixel: mi.Point2f) -> mi.Point2f:
        # film_width, film_height = self.film_size
        # return mi.Point2f(
        #     pixel.x / film_width,
        #     pixel.y / film_height)
        film_size = self.film().size()
        return pixel / mi.Point2f(film_size)

    def _sample_direction(self,
                          time: mi.Float,
                          sample: mi.Point2f,
                          active: mi.Mask) -> Tuple[mi.Float, mi.Vector3f]:
        origin = self._sensor_origin()

        if self.is_confocal:
            # Confocal sample always center of the pixel
            grid_sample = self._pixel_to_sample(self.confocal_pixel + 0.5)
            target = self.shape().sample_position(
                time, grid_sample, active
            ).p
        else:
            film_size = self.film().size()
            # instead of continuous samples over the whole shape,
            # discretize samples so they only land on the center of the film's
            # "pixels"
            grid_sample = self._pixel_to_sample(
                dr.floor(sample * film_size) + 0.5)
            target = self.shape().sample_position(
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
        # NOTE: all the parameters are set as NonDifferentiable by default
        super().traverse(callback)
        callback.put_parameter(
            "needs_sample_3", self.needs_sample_3, mi.ParamFlags.NonDifferentiable)
        callback.put_parameter("account_first_and_last_bounces",
                               self.account_first_and_last_bounces, mi.ParamFlags.NonDifferentiable)
        callback.put_parameter(
            "is_confocal", self.is_confocal, mi.ParamFlags.NonDifferentiable)
        callback.put_parameter(
            "laser_bounce_opl", self.laser_bounce_opl, mi.ParamFlags.NonDifferentiable)
        callback.put_parameter(
            "confocal_pixel", self.confocal_pixel, mi.ParamFlags.NonDifferentiable)

    def parameters_changed(self, keys):
        super().parameters_changed(keys)

    def to_string(self):
        string = f"{type(self).__name__}[\n"
        string += f"  laser_bounce_opl = {self.laser_bounce_opl}, \n"
        string += f"  account_first_and_last_bounces = {self.account_first_and_last_bounces}, \n"
        string += f"  is_confocal = {self.is_confocal}, \n"
        if self.is_confocal:
            string += f"  confocal_pixel = {self.confocal_pixel}, \n"
        string += f"  film = { indent(self.film()) }, \n"
        string += f"]"
        return string


mi.register_sensor('nlos_capture_meter', lambda props: NLOSCaptureMeter(props))
