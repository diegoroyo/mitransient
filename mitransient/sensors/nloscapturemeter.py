from __future__ import annotations  # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

from mitsuba import Log, LogLevel

from typing import Tuple


class NLOSCaptureMeter(mi.Sensor):
    """
    TODO docs
    """

    # NOTE(diego): we assume the rays start in a vacuum
    # this is reasonable for NLOS scenes, but this can be changed
    # in the future if we want to support other media
    IOR_BASE = 1

    def __init__(self, props: mi.Properties):
        super().__init__(props)

        # try to get the emitter associated with the NLOSCaptureMeter
        self.m_emitter = None
        for name in props.property_names():
            if not name.startswith('_arg_'):
                continue

            value = props.get(name)
            if value is None:
                continue
            if not isinstance(value, mi.Emitter):
                continue

            if self.m_emitter is not None:
                raise Exception(
                    'Only one emitter can be specified per NLOS capture meter.')

            self.m_emitter = value

        self.m_needs_sample_3: bool = False

        self.m_account_first_and_last_bounces: bool = \
            props.get('account_first_and_last_bounces', True)

        self.world_transform: mi.Transform4f = \
            mi.Transform4f.translation(
                props.get('sensor_origin', mi.Point3f(0)))

        self.m_is_confocal: bool = props.get('confocal', False)
        self.m_film_size: mi.Vector2u = self.film.size()
        if self.m_is_confocal:
            self.film.size().assign(mi.Vector2u(1, 1))
            self.film.crop_size().assign(mi.Vector2u(1, 1))

        self.m_laser_origin: mi.Point3f = \
            props.get('laser_origin', mi.Point3f(0))
        laser_lookat3_pixel = props.get('laser_lookat_pixel', mi.Point3f(-1))
        laser_lookat3_3d = props.get('laser_lookat_3d', mi.Point3f(0))
        self.m_laser_lookat_is_pixel: bool = \
            laser_lookat3_pixel.x > 0.0 and laser_lookat3_pixel.y > 0.0
        if self.m_laser_lookat_is_pixel:
            film_width, film_height = self._film_size()
            if laser_lookat3_pixel.x < 0.0 or film_width < laser_lookat3_pixel.x:
                Log(LogLevel.Warn, 'Laser lookat pixel (X postiion) is out of bounds')
            if laser_lookat3_pixel.y < 0.0 or film_height < laser_lookat3_pixel.y:
                Log(LogLevel.Warn, 'Laser lookat pixel (Y postiion) is out of bounds')
            if abs(laser_lookat3_pixel.z) > dr.epsilon:
                Log(LogLevel.Warn, 'Laser lookat pixel (Z postiion) should be zero')
            self.m_laser_lookat: mi.Point3f = laser_lookat3_pixel
        else:
            self.m_laser_lookat: mi.Point3f = laser_lookat3_3d

        if self.m_emitter is None:
            Log(LogLevel.Warn,
                'This sensor should include a (projector) emitter. '
                'Adding a default one, but this is probably not what '
                'you want.')
            props_film = mi.Properties('projector')
            self.m_emitter: mi.Emitter = \
                mi.PluginManager.create_object(props_film)

    def _film_size(self) -> Tuple[mi.UInt, mi.UInt]:
        return (self.m_film_size.x, self.m_film_size.y)

    def _sensor_origin(self) -> mi.Point3f:
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

        if self.m_is_confocal:
            target = self.m_laser_target
        else:
            film_width, film_height = self._film_size()
            # instead of continuous samples over the whole shape,
            # discretize samples so they only land on the center of the film's
            # "pixels"
            grid_sample = self.pixel_to_sample(
                mi.Point2f(
                    dr.floor(sample.x * film_width) + 0.5,
                    dr.floor(sample.y * film_height) + 0.5))
            target = self.m_shape.sample_position(
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
        scene_emitters.append(self.m_emitter)

    def set_shape(self, shape: mi.Shape):
        super().set_shape(shape)  # sets self.m_shape

        if self.m_laser_lookat_is_pixel:
            self.m_laser_target = self.m_shape.sample_position(
                0.0,
                self._pixel_to_sample(
                    mi.Point2f(
                        self.m_laser_lookat.x(),
                        self.m_laser_lookat.y()
                    ))).p  # sampled position of PositionSample3f
            Log(LogLevel.Info,
                'Laser is pointed to pixel (%d, %d), '
                'which equals to 3D point (%d, %d, %d)',
                self.m_laser_lookat.x(), self.m_laser_lookat.y(),
                self.m_laser_target.x(), self.m_laser_target.y(), self.m_laser_target.z())
        else:
            self.m_laser_target = self.m_laser_lookat
            Log(LogLevel.Info,
                'Laser is pointed to 3D point (%d, %d, %d)',
                self.m_laser_target.x(), self.m_laser_target.y(), self.m_laser_target.z())

        self.m_emitter.world_transform().assign(
            mi.Transform4f.look_at(
                origin=self.m_laser_origin,
                target=self.m_laser_target,
                up=mi.Vector3f(0, 1, 0)))

        self.m_laser_bounce_opl = dr.norm(
            self.m_laser_target - self.m_laser_origin) * self.IOR_BASE

    def traverse(self, callback: mi.TraversalCallback):
        # TODO not implemented (?)
        super().traverse(callback)

    def sample_ray_differential(
            self, time: mi.Float,
            wavelength_sample: mi.Float, position_sample: mi.Point2f, aperture_sample: mi.Point2f,
            active: mi.Bool = True) -> Tuple[mi.RayDifferential3f, mi.Color3f]:
        origin = self._sensor_origin()
        sensor_distance, direction = self._sample_direction(
            time, position_sample, active)

        wavelengths, wav_weight = self.sample_wavelengths(
            dr.zeros(mi.SurfaceInteraction3f), wavelength_sample, active)

        if not self.m_account_first_and_last_bounces:
            time -= self.m_laser_bounce_opl + sensor_distance * self.IOR_BASE

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
        return self.m_shape.pdf_direction(it, ds, active)

    def eval(self, si: mi.SurfaceInteraction3f, active: mi.Bool = True) -> mi.Spectrum:
        return dr.pi / self.m_shape.surface_area()

    def bbox(self) -> mi.BoundingBox3f:
        return self.m_shape.bbox()

    def is_nlos_sensor(self) -> bool:
        return True

    def to_string(self):
        # TODO update with the rest of parameters
        # m_shape, m_emitter from NLOSCaptureMeter, m_film from Sensor, etc.
        return f'{type(self).__name__}[laser_target = {self.m_laser_target},' \
               f' confocal = { self.m_is_confocal }]'


mi.register_sensor('NLOSCaptureMeter', lambda props: NLOSCaptureMeter(props))
