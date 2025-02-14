from __future__ import annotations as __annotations__  # Delayed parsing of type annotations

import mitsuba as mi
import drjit as dr
# import gc

from typing import Union, Any, Tuple

from mitsuba.ad.integrators.common import ADIntegrator  # type: ignore
from ..films.transient_hdr_film import TransientHDRFilm

class TransientADIntegrator(ADIntegrator):

    def __init__(self, props: mi.Properties):
        super().__init__(props)  # initialize props: max_depth and rr_depth

        self.camera_unwarp = props.get("camera_unwarp", False)
        # TODO (JORGE): remove these attributes
        _ = props.get("gaussian_stddev", 0.5)
        _ = props.get("temporal_filter", "")

    def render(self: mi.SamplingIntegrator,
               scene: mi.Scene,
               sensor: Union[int, mi.Sensor] = 0,
               seed: mi.UInt32 = 0,
               spp: int = 0,
               develop: bool = True,
               evaluate: bool = True) -> Tuple[mi.TensorXf, mi.TensorXf]:
        
        if not develop:
            raise Exception("develop=True must be specified when "
                            "invoking AD integrators")
        
        self.check_transient_(scene, sensor)

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(
                sensor=sensor,
                seed=seed,
                spp=spp,
                aovs=self.aov_names()
            )
            # TODO(JORGE): add extension to allow for bigger spp

            # Generate a set of rays starting at the sensor
            ray, weight, pos = self.sample_rays(scene, sensor, sampler)

            # Launch the Monte Carlo sampling process in primal mode
            L, valid, aovs, _ = self.sample(
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler,
                ray=ray,
                depth=mi.UInt32(0),
                δL=None,
                δaovs=None,
                state_in=None,
                active=mi.Bool(True),
                add_transient=self.add_transient_f(
                    film=film, pos=pos, ray_weight=weight, sample_scale=1.0 / spp
                )
            )

            # Prepare an ImageBlock as specified by the film
            block = film.steady.create_block()

            # Only use the coalescing feature when rendering enough samples
            block.set_coalesce(block.coalesce() and spp >= 4)

            # Accumulate into the image block
            ADIntegrator._splat_to_block(
                block, film, pos,
                value=L * weight,
                weight=1.0,
                alpha=dr.select(valid, mi.Float(1), mi.Float(0)),
                aovs=aovs,
                wavelengths=ray.wavelengths
            )

            # Explicitly delete any remaining unused variables
            del sampler, ray, weight, pos, L, valid

            # Perform the weight division and return an image tensor
            film.steady.put_block(block)
            steady_image, transient_image = film.develop()

            return steady_image, transient_image

    def render_forward(self: mi.SamplingIntegrator,
                       scene: mi.Scene,
                       params: Any,
                       sensor: Union[int, mi.Sensor] = 0,
                       seed: mi.UInt32 = 0,
                       spp: int = 0) -> mi.TensorXf:
        # TODO implement render_forward (either here or move this function to RBIntegrator)
        raise NotImplementedError(
            "Check https://github.com/mitsuba-renderer/mitsuba3/blob/1e513ef94db0534f54a884f2aeab7204f6f1e3ed/src/python/python/ad/integrators/common.py"
        )

    def render_backward(self: mi.SamplingIntegrator,
                    scene: mi.Scene,
                    params: Any,
                    grad_in: mi.TensorXf,
                    sensor: Union[int, mi.Sensor] = 0,
                    seed: mi.UInt32 = 0,
                    spp: int = 0) -> None:
        # TODO implement render_backward (either here or move this function to RBIntegrator)
        raise NotImplementedError(
            "Check https://github.com/mitsuba-renderer/mitsuba3/blob/1e513ef94db0534f54a884f2aeab7204f6f1e3ed/src/python/python/ad/integrators/common.py"
        )
    
    def add_transient_f(self, film: TransientHDRFilm, pos: mi.Vector2f, ray_weight: mi.Float, sample_scale: mi.Float):
        """
        Return a lambda function for saving transient samples.
        It pre-multiplies the sample scale.
        """
        return (
            lambda spec, distance, wavelengths, active: film.add_transient_data(
                pos, distance, wavelengths, spec * sample_scale, ray_weight, active
            )
        )
    
    def check_transient_(self, scene: mi.Scene, sensor: mi.Sensor):
        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        from mitransient.sensors.nloscapturemeter import NLOSCaptureMeter

        if isinstance(sensor, NLOSCaptureMeter):
            if self.camera_unwarp:
                raise AssertionError(
                    "camera_unwarp is not supported for NLOSCaptureMeter. "
                    "Use account_first_and_last_bounces in the NLOSCaptureMeter plugin instead."
                )
            
        del NLOSCaptureMeter

        from mitransient.films.transient_hdr_film import TransientHDRFilm

        if not isinstance(sensor.film(), TransientHDRFilm):
            raise AssertionError(
                "The film of the sensor must be of type transient_hdr_film"
            )

        del TransientHDRFilm

    def to_string(self):
        string = f'{type(self).__name__}[\n'
        string += f'  max_depth = {self.max_depth}, \n'
        string += f'  rr_depth = { self.rr_depth }\n'
        string += f']'
        return string