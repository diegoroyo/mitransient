# Delayed parsing of type annotations
from __future__ import annotations as __annotations__

import mitsuba as mi
import drjit as dr
# import gc

from typing import Union, Any, Tuple

from mitsuba.ad.integrators.common import ADIntegrator  # type: ignore
from ..films.transient_hdr_film import TransientHDRFilm


class TransientADIntegrator(ADIntegrator):
    r"""
    .. _integrator-transientadintegrator:

    Transient AD Integrator
    -----------------------

    Abstract base class for transient integrators in ``mitransient``.
    """

    def __init__(self, props: mi.Properties):
        super().__init__(props)  # initialize props: max_depth and rr_depth

        self.camera_unwarp = props.get("camera_unwarp", False)
        # FIXME document this and add to other integrators maybe
        self.discard_direct_light = props.get("discard_direct_light", False)
        # TODO (diego): Figure out how to move these parameters to filter properties
        _ = props.get("gaussian_stddev", 0.5)
        _ = props.get("temporal_filter", "")

    def prepare(self, scene, sensor, seed, spp, aovs):
        film = sensor.film()
        original_sampler = sensor.sampler()
        sampler = original_sampler.clone()

        if spp != 0:
            sampler.set_sample_count(spp)

        spp = sampler.sample_count()
        sampler.set_samples_per_wavefront(spp)

        film_size = film.crop_size()

        if film.sample_border():
            film_size += 2 * film.rfilter().border_size()

        wavefront_size = dr.prod(film_size) * spp
        film.prepare(aovs)

        if wavefront_size <= 2**32:
            sampler.seed(seed, wavefront_size)
            # Intentionally pass it on a list to mantain compatibility
            return [(sampler, spp)]

        # It is not possible to render more than 2^32 samples
        # in a single pass (32-bit integer)
        # We reduce it even further, to 2^26, to make the progress
        # bar update more frequently at the cost of more overhead
        # to create the kernels/etc (measured ~5% overhead)
        spp_per_pass = int((2**26 - 1) / dr.prod(film_size))
        if spp_per_pass == 0:
            raise Exception("Your film is too big. Please make it smaller.")

        # Split into max-size jobs (maybe add reminder at the end)
        needs_remainder = spp % spp_per_pass != 0
        num_passes = spp // spp_per_pass + 1 * needs_remainder

        sampler.set_sample_count(num_passes)
        sampler.set_samples_per_wavefront(num_passes)
        sampler.seed(seed, num_passes)
        seeds = mi.UInt32(sampler.next_1d() * 2**32)

        def sampler_per_pass(i):
            if needs_remainder and i == num_passes - 1:
                spp_per_pass_i = spp % spp_per_pass
            else:
                spp_per_pass_i = spp_per_pass
            sampler_clone = sensor.sampler().clone()
            sampler_clone.set_sample_count(spp_per_pass_i)
            sampler_clone.set_samples_per_wavefront(spp_per_pass_i)
            sampler_clone.seed(seeds[i], dr.prod(film_size) * spp_per_pass_i)
            return sampler_clone, spp_per_pass_i

        return [sampler_per_pass(i) for i in range(num_passes)]

    def render(self: mi.SamplingIntegrator,
               scene: mi.Scene,
               sensor: Union[int, mi.Sensor] = 0,
               seed: mi.UInt32 = 0,
               spp: int = 0,
               develop: bool = True,
               evaluate: bool = True,
               progress_callback: function = None) -> Tuple[mi.TensorXf, mi.TensorXf]:
        if not develop:
            raise Exception("develop=True must be specified when "
                            "invoking AD integrators")

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]
        film = sensor.film()

        self.check_transient_(scene, sensor)

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            samplers_spps = self.prepare(
                scene=scene,
                sensor=sensor,
                seed=seed,
                spp=spp,
                aovs=self.aov_names()
            )

            # need to re-add in case the spp parameter was set to 0
            # (spp was set through the xml file)
            total_spp = 0
            for _, spp_i in samplers_spps:
                total_spp += spp_i

            for i, (sampler_i, spp_i) in enumerate(samplers_spps):
                # Generate a set of rays starting at the sensor
                ray, weight, pos = self.sample_rays(scene, sensor, sampler_i)

                # Launch the Monte Carlo sampling process in primal mode
                L, valid, aovs, _ = self.sample(
                    mode=dr.ADMode.Primal,
                    scene=scene,
                    sampler=sampler_i,
                    ray=ray,
                    depth=mi.UInt32(0),
                    δL=None,
                    δaovs=None,
                    state_in=None,
                    active=mi.Bool(True),
                    add_transient=self.add_transient_f(
                        film=film, pos=pos, ray_weight=weight, sample_scale=1.0 / total_spp
                    )
                )

                # Prepare an ImageBlock as specified by the film
                block = film.steady.create_block()

                # Only use the coalescing feature when rendering enough samples
                block.set_coalesce(block.coalesce() and spp_i >= 4)

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
                del sampler_i, ray, weight, pos, L, valid

                # Perform the weight division and return an image tensor
                film.steady.put_block(block)

                # Report progress
                if progress_callback:
                    progress_callback((i + 1) / len(samplers_spps))

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
        from mitransient.films.phasor_hdr_film import PhasorHDRFilm

        if not (isinstance(sensor.film(), TransientHDRFilm) or isinstance(sensor.film(), PhasorHDRFilm)):
            raise AssertionError(
                "The film of the sensor must be of type transient_hdr_film or phasor_hdr_film"
            )

        del TransientHDRFilm

    def to_string(self):
        string = f'{type(self).__name__}[\n'
        string += f'  max_depth = {self.max_depth}, \n'
        string += f'  rr_depth = { self.rr_depth }\n'
        string += f']'
        return string
