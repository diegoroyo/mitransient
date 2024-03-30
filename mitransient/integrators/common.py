# Delayed parsing of type annotations
from __future__ import annotations as __annotations__

import mitsuba as mi
import drjit as dr
import gc

from typing import Union, Any, Callable, Optional, Tuple

from mitsuba.ad.integrators.common import ADIntegrator  # type: ignore


class TransientADIntegrator(ADIntegrator):
    """
    Abstract base class for transient integrators in Transient Mitsuba 3

    The automatic differentiation (AD) part is not supported at the moment,
    should be implemented in the render_forward and render_backward functions

    For further information:
    - https://github.com/diegoroyo/mitsuba3/blob/v3.3.0-nlos/src/python/python/ad/integrators/common.py
    """

    def __init__(self, props=mi.Properties()):
        super().__init__(props)

        # imported: max_depth and rr_depth

        # NOTE temporal_filter can take: box, gaussian, or an empty string
        # (which sets it to use the same temporal filter same as the film's rfilter)
        self.temporal_filter = props.get("temporal_filter", "")
        # FIXME these parameters can be set in other places, probably
        self.camera_unwarp = props.get("camera_unwarp", False)
        self.gaussian_stddev = props.get("gaussian_stddev", 0.5)

    def to_string(self):
        # TODO this should go in transientpath, transientnlospath, transient_prb_volpath.py
        return (
            f"{type(self).__name__}[max_depth = {self.max_depth},"
            f" rr_depth = { self.rr_depth }]"
        )

    def _prepare_los(
        self, sensor: mi.Sensor, seed: int = 0, spp: int = 0, aovs: list = []
    ):

        film = sensor.film()
        sampler = sensor.sampler().clone()

        if spp != 0:
            sampler.set_sample_count(spp)

        spp = sampler.sample_count()
        sampler.set_samples_per_wavefront(spp)

        film_size = film.crop_size()

        if film.sample_border():
            film_size += 2 * film.rfilter().border_size()

        wavefront_size = dr.prod(film_size) * spp

        if wavefront_size > 2**32:
            raise Exception(
                "The total number of Monte Carlo samples required by this "
                "rendering task (%i) exceeds 2^32 = 4294967296. Please use "
                "fewer samples per pixel or render using multiple passes."
                % wavefront_size
            )

        sampler.seed(seed, wavefront_size)
        film.prepare(aovs)

        return sampler, spp

    def _prepare_nlos(
        self, sensor: mi.Sensor, seed: int = 0, spp: int = 0, aovs: list = []
    ):

        film = sensor.film()
        sampler = sensor.sampler().clone()

        film_size = film.crop_size()
        if film.sample_border():
            film_size += 2 * film.rfilter().border_size()
        film.prepare(aovs)

        if spp == 0:
            spp = sampler.sample_count()

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

    def prepare(self, sensor: mi.Sensor, seed: int = 0, spp: int = 0, aovs: list = []):
        """
        Given a sensor and a desired number of samples per pixel, this function
        computes the necessary number of Monte Carlo samples and then suitably
        seeds the sampler underlying the sensor.

        Returns the created sampler and the final number of samples per pixel
        (which may differ from the requested amount depending on the type of
        ``Sampler`` being used)

        Parameter ``sensor`` (``int``, ``mi.Sensor``):
            Specify a sensor to render the scene from a different viewpoint.

        Parameter ``seed` (``int``)
            This parameter controls the initialization of the random number
            generator during the primal rendering step. It is crucial that you
            specify different seeds (e.g., an increasing sequence) if subsequent
            calls should produce statistically independent images (e.g. to
            de-correlate gradient-based optimization steps).

        Parameter ``spp`` (``int``):
            Optional parameter to override the number of samples per pixel for the
            primal rendering step. The value provided within the original scene
            specification takes precedence if ``spp=0``.
        """
        from mitransient.sensors.nloscapturemeter import NLOSCaptureMeter

        if isinstance(sensor, NLOSCaptureMeter):
            return self._prepare_nlos(sensor, seed, spp, aovs)
        else:
            return [self._prepare_los(sensor, seed, spp, aovs)]

    def prepare_transient(self, scene: mi.Scene, sensor: mi.Sensor):
        """
        Prepare the integrator to perform a transient simulation
        """
        import numpy as np

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        from mitransient.sensors.nloscapturemeter import NLOSCaptureMeter

        if isinstance(sensor, NLOSCaptureMeter):
            if self.camera_unwarp:
                raise AssertionError(
                    "camera_unwarp is not supported for NLOSCaptureMeter. "
                    "Use account_first_and_last_bounces in the NLOSCaptureMeter plugin instead."
                )

        film = sensor.film()
        from mitransient.films.transient_hdr_film import TransientHDRFilm

        if not isinstance(film, TransientHDRFilm):
            raise AssertionError(
                "The film of the sensor must be of type transient_hdr_film"
            )

        # Create the transient block responsible for storing the time contribution
        crop_size = film.crop_size()
        temporal_bins = film.temporal_bins
        size = np.array([crop_size.x, crop_size.y, temporal_bins])

        def load_filter(name, **kargs):
            """
            Shorthand for loading an specific reconstruction kernel
            """
            kargs["type"] = name
            f = mi.load_dict(kargs)
            return f

        def get_filters(sensor):
            """
            Selecting the temporal reconstruction filter.
            """
            if self.temporal_filter == "box":
                time_filter = load_filter("box")
            elif self.temporal_filter == "gaussian":
                stddev = max(self.gaussian_stddev, 0.5)
                time_filter = load_filter("gaussian", stddev=stddev)
            else:
                time_filter = sensor.film().rfilter()

            return [sensor.film().rfilter(), sensor.film().rfilter(), time_filter]

        filters = get_filters(sensor)
        film.prepare_transient(size=size, rfilter=filters)
        self._film = film

    def add_transient_f(self, pos, ray_weight, sample_scale):
        """
        Return a lambda function for saving transient samples.
        It pre-multiplies the sample scale.
        """
        return (
            lambda spec, distance, wavelengths, active: self._film.add_transient_data(
                spec * sample_scale, distance, wavelengths, active, pos, ray_weight
            )
        )

    def render(
        self: mi.SamplingIntegrator,
        scene: mi.Scene,
        sensor: Union[int, mi.Sensor] = 0,
        seed: int = 0,
        spp: int = 0,
        develop: bool = True,
        evaluate: bool = True,
        progress_callback: function = None,
    ) -> mi.TensorXf:

        if not develop:
            raise Exception(
                "develop=True must be specified when " "invoking AD integrators"
            )

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            prepare_result = self.prepare(
                sensor=sensor, seed=seed, spp=spp, aovs=self.aov_names()
            )

            total_spp = 0
            for sampler, spp in prepare_result:
                total_spp += spp

            for i, (sampler, spp) in enumerate(prepare_result):
                # Generate a set of rays starting at the sensor
                ray, ray_weight, pos = self.sample_rays(scene, sensor, sampler)

                # Launch the Monte Carlo sampling process in primal mode
                L, valid, state = self.sample(
                    mode=dr.ADMode.Primal,
                    scene=scene,
                    sampler=sampler,
                    ray=ray,
                    depth=mi.UInt32(0),
                    δL=None,
                    state_in=None,
                    reparam=None,
                    active=mi.Bool(True),
                    max_distance=self._film.end_opl(),
                    add_transient=self.add_transient_f(
                        pos=pos, ray_weight=ray_weight, sample_scale=1.0 / total_spp
                    ),
                )

                # Prepare an ImageBlock as specified by the film
                block = film.steady.create_block()

                # Only use the coalescing feature when rendering enough samples
                block.set_coalesce(block.coalesce() and spp >= 4)

                # Accumulate into the image block
                alpha = dr.select(valid, mi.Float(1), mi.Float(0))
                if mi.has_flag(film.steady.flags(), mi.FilmFlags.Special):
                    aovs = film.steady.prepare_sample(
                        L * ray_weight,
                        ray.wavelengths,
                        block.channel_count(),
                        alpha=alpha,
                    )
                    block.put(pos, aovs)
                    del aovs
                else:
                    block.put(pos, ray.wavelengths, L * ray_weight, alpha)

                # Explicitly delete any remaining unused variables
                del sampler, ray, ray_weight, pos, L, valid, alpha
                gc.collect()

                # Perform the ray_weight division and return an image tensor
                film.steady.put_block(block)
                if progress_callback:
                    progress_callback((i + 1) / len(prepare_result))

            self.primal_image = film.steady.develop()
            transient_image = film.transient.develop()

            return self.primal_image, transient_image

    def render_forward(
        self: mi.SamplingIntegrator,
        scene: mi.Scene,
        params: Any,
        sensor: Union[int, mi.Sensor] = 0,
        seed: int = 0,
        spp: int = 0,
    ) -> mi.TensorXf:
        # TODO implement render_forward (either here or move this function to RBIntegrator)
        raise NotImplementedError(
            "Check https://github.com/mitsuba-renderer/mitsuba3/blob/1e513ef94db0534f54a884f2aeab7204f6f1e3ed/src/python/python/ad/integrators/common.py"
        )

    def render_backward(
        self: mi.SamplingIntegrator,
        scene: mi.Scene,
        params: Any,
        grad_in: mi.TensorXf,
        sensor: Union[int, mi.Sensor] = 0,
        seed: int = 0,
        spp: int = 0,
    ) -> None:
        # TODO implement render_backward (either here or move this function to RBIntegrator)
        raise NotImplementedError(
            "Check https://github.com/mitsuba-renderer/mitsuba3/blob/1e513ef94db0534f54a884f2aeab7204f6f1e3ed/src/python/python/ad/integrators/common.py"
        )

    # NOTE(diego): only change is the addition of the add_transient argument
    def sample(
        self,
        mode: dr.ADMode,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.Ray3f,
        depth: mi.UInt32,
        δL: Optional[mi.Spectrum],
        state_in: Any,
        reparam: Optional[
            Callable[[mi.Ray3f, mi.UInt32, mi.Bool],
                     Tuple[mi.Vector3f, mi.Float]]
        ],
        active: mi.Bool,
        add_transient,
    ) -> Tuple[mi.Spectrum, mi.Bool]:
        """
        This function does the main work of differentiable rendering and
        remains unimplemented here. It is provided by subclasses of the
        ADIntegrator interface.

        This is exactly the same as non-transient ADIntegrator,
        but now includes an add_transient parameter to store the time-resolved
        contributions of light.

        References:
        - https://github.com/diegoroyo/mitsuba3/blob/61c7cd1cff1937b2a041f1eacd90205b8e7e8c4a/src/python/python/ad/integrators/common.py#L489
        """

        raise Exception(
            "ADIntegrator does not provide the sample() method. "
            "It should be implemented by subclasses that "
            "specialize the abstract ADIntegrator interface."
        )


def mis_weight(pdf_a, pdf_b):
    """
    Compute the Multiple Importance Sampling (MIS) weight given the densities
    of two sampling strategies according to the power heuristic.
    """
    a2 = dr.sqr(pdf_a)
    b2 = dr.sqr(pdf_b)
    w = a2 / (a2 + b2)
    return dr.detach(dr.select(dr.isfinite(w), w, 0))
