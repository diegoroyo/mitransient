# Delayed parsing of type annotations
from __future__ import annotations as __annotations__

import mitsuba as mi
import drjit as dr
# import gc

from typing import Union, Any, Tuple, Sequence

from mitsuba import Log, LogLevel
from mitsuba.ad.integrators.common import ADIntegrator  # type: ignore
from ..films.transient_hdr_film import TransientHDRFilm
from ..utils import β_init
from ..version import Version


class TransientADIntegrator(ADIntegrator):
    """
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

    def _splat_to_block(self,
                        block: mi.ImageBlock,
                        film: mi.Film,
                        pos: mi.Point2f,
                        value: mi.Spectrum,
                        weight: mi.Float,
                        alpha: mi.Float,
                        aovs: Sequence[mi.Float],
                        wavelengths: mi.Spectrum):
        # FIXME: Wrapper function to avoid calling mitsuba/python/ad/common.py: _splat_to_block, which fails with
        #       the transient polarimetric integrator. The function is fixed in mitsuba 3.7, the wrapper function will
        #       not be needed when mitransient is updated to work with mitsuba 3.7.
        '''Helper function to splat values to a imageblock'''
        if (dr.all(mi.has_flag(film.flags(), mi.FilmFlags.Special))):
            aovs = film.prepare_sample(value, wavelengths,
                                       block.channel_count(),
                                       weight=weight,
                                       alpha=alpha)
            block.put(pos, aovs)
            del aovs
        else:
            if mi.is_polarized:
                value = mi.unpolarized_spectrum(value)
            if mi.is_spectral:
                rgb = mi.spectrum_to_srgb(value, wavelengths)
            elif mi.is_monochromatic:
                rgb = mi.Color3f(value.x)
            else:
                rgb = value
            if mi.has_flag(film.flags(), mi.FilmFlags.Alpha):
                aovs = [rgb.x, rgb.y, rgb.z, alpha, weight] + aovs
            else:
                aovs = [rgb.x, rgb.y, rgb.z, weight] + aovs
            block.put(pos, aovs)

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
                    β=β_init(sensor, ray),
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

                # NOTE(diego): Mitsuba 3.6.X needs extra care when dealing
                # with polarized functions, so we'll our version instead
                splat_function = (
                    ADIntegrator._splat_to_block
                    if Version(mi.__version__) >= Version('3.7.0')
                    else self._splat_to_block
                )
                # Accumulate into the image block
                splat_function(
                    block, film, pos,
                    value=L * mi.Spectrum(weight),
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
                aovs=[],
            )

            if len(samplers_spps) > 1:
                Log(LogLevel.ERROR,
                    "render_forward is not implemented for >2^26 samples."
                    "Please set spp to a lower value or reduce the dimensions of your image.")

            # need to re-add in case the spp parameter was set to 0
            # (spp was set through the xml file)
            total_spp = 0
            for _, spp_i in samplers_spps:
                total_spp += spp_i

            for sampler_i, spp_i in samplers_spps:
                # Generate a set of rays starting at the sensor
                ray, weight, pos = self.sample_rays(scene, sensor, sampler_i)

                # Launch the Monte Carlo sampling process in primal mode
                # NOTE(diego): we only do this to get state_out to pass it to the function below
                # we dont care about add_transient
                _, __, ___, state_out = self.sample(
                    mode=dr.ADMode.Primal,
                    scene=scene,
                    sampler=sampler_i.clone(),
                    ray=ray,
                    depth=mi.UInt32(0),
                    β=β_init(sensor, ray),
                    δL=None,
                    δaovs=None,
                    state_in=None,
                    active=mi.Bool(True),
                    add_transient=lambda _, __, ___, ____: None
                )

                # Launch the Monte Carlo sampling process in backward AD mode
                # NOTE: the integrator expects δL to have shape (num_pixels, num_spp),
                # but in the transient domain it has shape (num_pixels * num_time_bins, num_spp)
                # so the integrator provides a read_δL function that reads the corresponding time bin
                # for each pixel and gives a read_δL with shape (num_pixels, num_spp)
                # Launch the Monte Carlo sampling process in primal mode
                δL, valid, δaovs, _ = self.sample(
                    mode=dr.ADMode.Forward,
                    scene=scene,
                    sampler=sampler_i,
                    ray=ray,
                    depth=mi.UInt32(0),
                    β=β_init(sensor, ray),
                    δL=None,
                    δaovs=None,
                    state_in=state_out,
                    active=mi.Bool(True),
                    add_transient=self.add_transient_f(
                        film=film, pos=pos, ray_weight=weight, sample_scale=1.0 / total_spp
                    )
                )

                # Prepare an ImageBlock as specified by the film
                block = film.steady.create_block()

                # Only use the coalescing feature when rendering enough samples
                block.set_coalesce(block.coalesce() and spp_i >= 4)

                # NOTE(diego): Mitsuba 3.6.X needs extra care when dealing
                # with polarized functions, so we'll our version instead
                splat_function = (
                    ADIntegrator._splat_to_block
                    if Version(mi.__version__) >= Version('3.7.0')
                    else self._splat_to_block
                )
                # Accumulate into the image block
                splat_function(
                    block, film, pos,
                    value=δL * mi.Spectrum(weight),
                    weight=1.0,
                    alpha=dr.select(valid, mi.Float(1), mi.Float(0)),
                    aovs=δaovs,
                    wavelengths=ray.wavelengths
                )

                # Explicitly delete any remaining unused variables
                del sampler_i, ray, weight, pos, δL, valid

                # Perform the weight division and return an image tensor
                film.steady.put_block(block)

            steady_image, transient_image = film.develop()
            return steady_image, transient_image

    def render_backward(self: mi.SamplingIntegrator,
                        scene: mi.Scene,
                        params: Any,
                        grad_in: mi.TensorXf,
                        sensor: Union[int, mi.Sensor] = 0,
                        seed: mi.UInt32 = 0,
                        spp: int = 0) -> None:
        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]
        film = sensor.film()
        self.check_transient_(scene, sensor)

        grad_in_steady, grad_in_transient = grad_in

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            samplers_spps = self.prepare(
                scene=scene,
                sensor=sensor,
                seed=seed,
                spp=spp,
                aovs=[],
            )

            if len(samplers_spps) > 1:
                Log(LogLevel.ERROR,
                    "render_backward is not implemented for >2^26 samples."
                    "Please set spp to a lower value or reduce the dimensions of your image.")

            for sampler_i, spp_i in samplers_spps:
                # Generate a set of rays starting at the sensor
                ray, weight, pos = self.sample_rays(scene, sensor, sampler_i)

                # NOTE(diego): This assumes that the user uses a box filter
                # (i.e. rays only affect the pixel they are in)
                # otherwise you should use splatting_and_backward_gradient_image
                # from the original mitsuba3 code
                h, w, t, c = grad_in_transient.shape
                δL = grad_in_transient + \
                    dr.reshape(grad_in_steady, (h, w, 1, c))
                δL = dr.reshape(dr.moveaxis(δL, -1, 0), (c, h*w*t))

                # Launch the Monte Carlo sampling process in primal mode
                # NOTE(diego): we only do this to get state_out to pass it to the function below
                # we dont care about add_transient
                _, __, ___, state_out = self.sample(
                    mode=dr.ADMode.Primal,
                    scene=scene,
                    sampler=sampler_i.clone(),
                    ray=ray,
                    depth=mi.UInt32(0),
                    β=β_init(sensor, ray),
                    δL=None,
                    δaovs=None,
                    state_in=None,
                    active=mi.Bool(True),
                    add_transient=lambda _, __, ___, ____: None
                )

                # Launch the Monte Carlo sampling process in backward AD mode
                # NOTE: the integrator expects δL to have shape (num_pixels, num_spp),
                # but in the transient domain it has shape (num_pixels * num_time_bins, num_spp)
                # so the integrator provides a read_δL function that reads the corresponding time bin
                # for each pixel and gives a read_δL with shape (num_pixels, num_spp)
                _ = self.sample(
                    mode=dr.ADMode.Backward,
                    scene=scene,
                    sampler=sampler_i,
                    ray=ray,
                    depth=mi.UInt32(0),
                    β=β_init(sensor, ray),
                    δL=δL,
                    δaovs=None,
                    state_in=state_out,
                    active=mi.Bool(True),
                    add_transient=lambda _, __, ___, ____: None,
                    gather_derivatives_at_distance=lambda δL, distance:
                        film.gather_derivatives_at_distance(pos, δL, distance)
                )

                del ray, weight, pos, sampler_i
                dr.eval()

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
