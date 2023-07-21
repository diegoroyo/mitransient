from __future__ import annotations  # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

from mitsuba import Log, LogLevel

# NOTE(diego): idk why, but mitsuba's variant is not set for NLOSCaptureMeter.__init__ method
# it is fine if we import it here
from mitsuba import Emitter, Float, Transform4f

from mitsuba import ScalarVector2u
from drjit.scalar import Array3f as ScalarArray3f  # type: ignore

from typing import Tuple


class NLOSCaptureMeter(mi.Sensor):
    """
    TODO(diego) add docs
    """

    # NOTE(diego): we assume the rays start in a vacuum
    # this is reasonable for NLOS scenes, but this can be changed
    # in the future if we want to support other media
    IOR_BASE = 1

    def __init__(self, props: mi.Properties):
        super().__init__(props)

        # try to get the emitter associated with the NLOSCaptureMeter
        self.emitter = None
        for name in props.property_names():
            if not name.startswith('_arg_'):
                continue

            value = props.get(name)
            if value is None:
                continue
            if not isinstance(value, Emitter):
                continue

            if self.emitter is not None:
                raise Exception(
                    'Only one emitter can be specified per NLOS capture meter.')

            self.emitter = value

        if self.emitter is None:
            raise Exception(
                'No laser emitter was specified for the NLOS capture meter.')

        self.needs_sample_3: bool = False

        self.account_first_and_last_bounces: bool = \
            props.get('account_first_and_last_bounces', True)

        self.world_transform: Transform4f = \
            Transform4f.translate(
                props.get('sensor_origin', ScalarArray3f(0)))

        self.is_confocal: bool = props.get('confocal', False)
        self.film_size: ScalarVector2u = self.film().size()
        if self.is_confocal:
            # NOTE: uses custom python interface for c++'s set_size method
            self.film().set_size(ScalarVector2u(1, 1))

        self.laser_origin: ScalarArray3f = props.get(
            'laser_origin', ScalarArray3f(0))
        laser_lookat3_pixel = props.get(
            'laser_lookat_pixel', ScalarArray3f(-1))
        laser_lookat3_3d = props.get('laser_lookat_3d', ScalarArray3f(0))
        self.laser_lookat_is_pixel: bool = \
            laser_lookat3_pixel.x > 0.0 and laser_lookat3_pixel.y > 0.0
        if self.laser_lookat_is_pixel:
            film_width, film_height = self.film_size
            if laser_lookat3_pixel.x < 0.0 or film_width < laser_lookat3_pixel.x:
                Log(LogLevel.Warn, 'Laser lookat pixel (X postiion) is out of bounds')
            if laser_lookat3_pixel.y < 0.0 or film_height < laser_lookat3_pixel.y:
                Log(LogLevel.Warn, 'Laser lookat pixel (Y postiion) is out of bounds')
            if dr.abs(laser_lookat3_pixel.z) > dr.epsilon(Float):
                Log(LogLevel.Warn, 'Laser lookat pixel (Z postiion) should be zero')
            self.laser_lookat: ScalarArray3f = laser_lookat3_pixel
        else:
            self.laser_lookat: ScalarArray3f = laser_lookat3_3d
        self.laser_target = None

    def _sensor_origin(self) -> ScalarArray3f:
        return self.world_transform.translation()

    def _pixel_to_sample(self, pixel: mi.Point2f) -> mi.Point2f:
        film_width, film_height = self.film_size
        return mi.Point2f(
            pixel.x / film_width,
            pixel.y / film_height)

    def _sample_direction(self,
                          time: mi.Float,
                          sample: mi.Point2f,
                          active: mi.Mask) -> Tuple[mi.Float, mi.Vector3f]:
        origin = self._sensor_origin()

        if self.is_confocal:
            target = self.laser_target
        else:
            film_width, film_height = self.film_size
            # instead of continuous samples over the whole shape,
            # discretize samples so they only land on the center of the film's
            # "pixels"
            grid_sample = self._pixel_to_sample(
                mi.Point2f(
                    dr.floor(sample.x * film_width) + 0.5,
                    dr.floor(sample.y * film_height) + 0.5))
            target = self.shape().sample_position(
                time, grid_sample, active
            ).p  # sampled position of PositionSample3f

        direction = target - origin
        distance = dr.norm(direction)
        direction /= distance
        return distance, direction

    def set_scene(self, scene: mi.Scene):
        # NOTE uses custom add_emitter function
        self.emitter.set_scene(scene)
        scene.add_emitter(self.emitter)
        scene_emitters = scene.emitters()
        if len(scene_emitters) > 1:
            Log(LogLevel.Warn,
                f'You have defined multiple ({len(scene_emitters)}) emitters in the scene with a NLOS capture meter.')

    def set_shape(self, shape: mi.Shape):
        # NOTE: set_shape is called by the integrator
        # super().set_shape(shape)  # sets self.shape

        if self.laser_lookat_is_pixel:
            # NOTE probably this can be made easier
            self.laser_target = ScalarArray3f(dr.ravel(
                self.shape().sample_position(
                    0.0,
                    self._pixel_to_sample(
                        mi.Point2f(
                            self.laser_lookat.x,
                            self.laser_lookat.y
                        ))).p))  # sampled position of PositionSample3f
            Log(LogLevel.Info,
                'Laser is pointed to pixel ({sx:.5f}, {sy:.5f}), '
                'which equals to 3D point ({px:.5f}, {py:.5f}, {pz:.5f})'.format(
                    sx=self.laser_lookat.x, sy=self.laser_lookat.y,
                    px=self.laser_target.x, py=self.laser_target.y, pz=self.laser_target.z))
        else:
            self.laser_target = self.laser_lookat
            Log(LogLevel.Info,
                'Laser is pointed to 3D point ({px:.5f}, {py:.5f}, {pz:.5f})'.format(
                    px=self.laser_target.x, py=self.laser_target.y, pz=self.laser_target.z))

        # NOTE uses custom set_world_transform function
        lookat = mi.Transform4f.look_at(
            origin=self.laser_origin,
            target=self.laser_target,
            up=mi.Vector3f(0, 1, 0))
        self.emitter.set_world_transform(lookat)

        self.laser_bounce_opl = dr.norm(
            self.laser_target - self.laser_origin) * self.IOR_BASE

    def sample_ray_differential(
            self, time: mi.Float,
            sample1: mi.Float, sample2: mi.Point2f, sample3: mi.Point2f,
            active: mi.Bool = True) -> Tuple[mi.RayDifferential3f, mi.Color3f]:
        origin = self._sensor_origin()
        sensor_distance, direction = self._sample_direction(
            time, sample2, active)

        # TODO call to sample_wavelengths yields
        # *** RuntimeError: Tried to call pure virtual function "Sensor::sample_wavelengths"
        # but it is only used in spectral mode (https://github.com/mitsuba-renderer/mitsuba3/blob/ff9cf94323703885068779b15be36345a2eadb89/include/mitsuba/core/spectrum.h#L471)
        # we do not have support for spectral rendering anyway, transient block is stuck with 3 channels I think
        wavelengths, wav_weight = [], 1
        # wavelengths, wav_weight = self.sample_wavelengths(
        #     dr.zeros(mi.SurfaceInteraction3f), sample1, active)

        if not self.account_first_and_last_bounces:
            time -= self.laser_bounce_opl + sensor_distance * self.IOR_BASE

        return (
            mi.RayDifferential3f(origin, direction, time, wavelengths),
            mi.unpolarized_spectrum(wav_weight) * dr.pi
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
        # TODO: all the parameters are set as NonDifferentiable by default
        super().traverse(callback)
        callback.put_object("emitter", self.emitter, mi.ParamFlags.NonDifferentiable)
        callback.put_parameter("needs_sample_3", self.needs_sample_3, mi.ParamFlags.NonDifferentiable)
        callback.put_parameter("account_first_and_last_bounces", self.account_first_and_last_bounces, mi.ParamFlags.NonDifferentiable)
        callback.put_parameter("is_confocal", self.is_confocal, mi.ParamFlags.NonDifferentiable)
        callback.put_parameter("film_size", self.film_size, mi.ParamFlags.NonDifferentiable)
        callback.put_parameter("laser_origin", self.laser_origin, mi.ParamFlags.NonDifferentiable)
        callback.put_parameter("laser_lookat", self.laser_lookat, mi.ParamFlags.NonDifferentiable)
        callback.put_parameter("laser_target", self.laser_target, mi.ParamFlags.NonDifferentiable)
        callback.put_parameter("laser_bounce_opl", self.laser_bounce_opl, mi.ParamFlags.NonDifferentiable)

    def parameters_changed(self, keys):
        super().parameters_changed(keys)

    def to_string(self):
        # TODO(diego) update with the rest of parameters
        # m_shape, m_emitter from NLOSCaptureMeter, m_film from Sensor, etc.
        string = f"{type(self).__name__}[\n"
        string += f"  needs_sample_3 = {self.needs_sample_3},"
        string += f"  account_first_and_last_bounces = {self.account_first_and_last_bounces},"
        string += f"  is_confocal = {self.is_confocal},"
        string += f"  film_size = {self.film_size},"
        string += f"  laser_origin = {self.laser_origin},"
        string += f"  laser_lookat = {self.laser_lookat},"
        string += f"  laser_target = {self.laser_target},"
        string += f"  laser_bounce_opl = {self.laser_bounce_opl},"
        string += f"]"
        return string


mi.register_sensor('nlos_capture_meter', lambda props: NLOSCaptureMeter(props))
