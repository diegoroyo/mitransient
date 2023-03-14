from __future__ import annotations  # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

from mitsuba import Log, LogLevel

# NOTE(diego): idk why, but mitsuba's variant is not set for NLOSCaptureMeter.__init__ method
# it is fine if we import it here
from mitsuba import Emitter, Float, Transform4f, Properties, PluginManager

from mitsuba import ScalarVector2u
from drjit.scalar import Array2f as ScalarArray2f, Array3f as ScalarArray3f

from typing import Tuple


class NLOSCaptureMeter(mi.Sensor):
    """
    FIXME(diego) add docs
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

        self.needs_sample_3: bool = False

        self.account_first_and_last_bounces: bool = \
            props.get('account_first_and_last_bounces', True)

        self.world_transform: Transform4f = \
            Transform4f.translate(
                props.get('sensor_origin', ScalarArray3f(0)))

        self.is_confocal: bool = props.get('confocal', False)
        self.film_size: ScalarVector2u = self.film().size()
        if self.is_confocal:
            self.film().size().assign(ScalarVector2u(1, 1))
            self.film().crop_size().assign(ScalarVector2u(1, 1))

        self.laser_origin: ScalarArray3f = \
            props.get('laser_origin', ScalarArray3f(0))
        laser_lookat3_pixel = props.get(
            'laser_lookat_pixel', ScalarArray3f(-1))
        laser_lookat3_3d = props.get('laser_lookat_3d', ScalarArray3f(0))
        self.laser_lookat_is_pixel: bool = \
            laser_lookat3_pixel.x > 0.0 and laser_lookat3_pixel.y > 0.0
        if self.laser_lookat_is_pixel:
            film_width, film_height = self._film_size()
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

        if self.emitter is None:
            Log(LogLevel.Warn,
                'This sensor should include a (projector) emitter. '
                'Adding a default one, but this is probably not what '
                'you want.')
            props_film = Properties('projector')
            self.emitter: Emitter = \
                PluginManager.create_object(props_film)

    def _film_size(self) -> Tuple[mi.UInt, mi.UInt]:
        return (self.film_size.x, self.film_size.y)

    def _sensor_origin(self) -> ScalarArray3f:
        return self.world_transform.translation()

    def _pixel_to_sample(self, pixel: mi.Point2f) -> mi.Point2f:
        film_width, film_height = self._film_size()
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
            film_width, film_height = self._film_size()
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
        scene_emitters = scene.emitters()
        if len(scene_emitters) > 0:
            Log(LogLevel.Warn,
                'This sensor only supports exactly one emitter object, '
                'which should be placed inside the sensor. Please remove '
                'other emitters.')
        scene_emitters.append(self.emitter)

    def set_shape(self, shape: mi.Shape):
        # super().set_shape(shape)  # sets self.shape

        if self.laser_lookat_is_pixel:
            # FIXME probably this can be made easier
            self.laser_target = ScalarArray3f(dr.ravel(
                self.shape().sample_position(
                    0.0,
                    self._pixel_to_sample(
                        mi.Point2f(
                            self.laser_lookat.x,
                            self.laser_lookat.y
                        ))).p))  # sampled position of PositionSample3f
            Log(LogLevel.Warn,
                'Laser is pointed to pixel ({sx:.5f}, {sy:.5f}), '
                'which equals to 3D point ({px:.5f}, {py:.5f}, {pz:.5f})'.format(
                    sx=self.laser_lookat.x, sy=self.laser_lookat.y,
                    px=self.laser_target.x, py=self.laser_target.y, pz=self.laser_target.z))
        else:
            self.laser_target = self.laser_lookat
            Log(LogLevel.Info,
                'Laser is pointed to 3D point ({px:.5f}, {py:.5f}, {pz:.5f})'.format(
                    px=self.laser_target.x, py=self.laser_target.y, pz=self.laser_target.z))

        self.emitter.world_transform().assign(
            mi.Transform4f.look_at(
                origin=self.laser_origin,
                target=self.laser_target,
                up=mi.Vector3f(0, 1, 0)))

        self.laser_bounce_opl = dr.norm(
            self.laser_target - self.laser_origin) * self.IOR_BASE

    def traverse(self, callback: mi.TraversalCallback):
        # TODO not implemented (?)
        super().traverse(callback)

    def sample_ray_differential(
            self, time: mi.Float,
            sample1: mi.Float, sample2: mi.Point2f, sample3: mi.Point2f,
            active: mi.Bool = True) -> Tuple[mi.RayDifferential3f, mi.Color3f]:
        origin = self._sensor_origin()
        sensor_distance, direction = self._sample_direction(
            time, sample2, active)

        # FIXME call to sample_wavelengths yields
        # *** RuntimeError: Tried to call pure virtual function "Sensor::sample_wavelengths"
        # but it is only used in spectral mode (https://github.com/mitsuba-renderer/mitsuba3/blob/ff9cf94323703885068779b15be36345a2eadb89/include/mitsuba/core/spectrum.h#L471)
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

    def to_string(self):
        # FIXME(diego) update with the rest of parameters
        # m_shape, m_emitter from NLOSCaptureMeter, m_film from Sensor, etc.
        return f'{type(self).__name__}[laser_target = {self.laser_target},' \
               f' confocal = { self.is_confocal }]'


mi.register_sensor('nlos_capture_meter', lambda props: NLOSCaptureMeter(props))
