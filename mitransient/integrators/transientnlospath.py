from __future__ import annotations  # Delayed parsing of type annotations
from typing import Optional, Tuple, List, Callable

import drjit as dr
import mitsuba as mi
from mitsuba import Log, LogLevel
from mitsuba.ad.integrators.common import mis_weight  # type: ignore

from mitransient.integrators.common import TransientADIntegrator


class TransientNLOSPath(TransientADIntegrator):
    r"""
    .. _integrator-transient_nlos_path:

    Transient NLOS Path (:monosp:`transient_nlos_path`)
    ---------------------------------------------------

    Standard path tracing algorithm which now includes the time dimension,
    and *contains multiple sampling routines specific to non-line-of-sight (NLOS)
    scenes*. To render LOS scenes, use the `transient_path` integrator.
    Choose one or the other depending on if you have a LOS or NLOS scene.

    Based on: [Royo2022] Royo, D., García, J., Muñoz, A., & Jarabo, A. (2022).
    Non-line-of-sight transient rendering. Computers & Graphics, 107, 84-92.

    .. pluginparameters::

     * - filter_bounces
       - |int|
       - Only account for paths of specific number of bounces in the result.
         A value of 1 will only render single-bounce (direct-only) illumination
         3 will lead to three-bounce (single-corner) illumination in NLOS setups
         And so on. A value of -1 disables this feature (default: -1 i.e. disabled)

     * - discard_direct_paths
       - |bool|
       - If True, paths with only 1 bounce (direct illuminations) are discarded.
         If False, this parameter does not have any effect. (default: false)

     * - nlos_laser_sampling
       - |bool|
       - If False, lights are sampled using Next-Event Estimation.
         If True, lights are sampled using the Laser Sampling technique.
         See [Royo2022] for more information about Laser Sampling. (default: false)

     * - nlos_hidden_geometry_sampling
       - |bool|
       - If False, ray directions are sampled using material properties.
         If True, ray directions are sampled using the Hidden Geometry Sampling technique.
         See [Royo2022] for more information about Hidden Geometry Sampling (default: false)

     * - nlos_hidden_geometry_sampling_do_rroulette
       - |bool|
       - Only relevant when `nlos_hidden_geometry_sampling` is True.
         If False, always uses the Hidden Geometry Sampling technique
         to sample new directions.
         If True, uses Russian Roulette to choose between 50% Hidden Geometry Sampling and
         50% Material Sampling.
         See [Royo2022] for more information about Hidden Geometry Sampling (default: false)

     * - nlos_hidden_geometry_sampling_includes_relay_wall
       - |bool|
       - Only relevant when `nlos_hidden_geometry_sampling` is True.
         If False, points in the relay wall cannot be sampled using the
         Hidden Geometry Sampling technique.
         If True, points in the relay wall can be sampled.
         See [Royo2022] for more information about Hidden Geometry Sampling (default: false)

     * - temporal_filter
       - |string|
       - Can be either:
         - 'box' for a box filter (no parameters)
         - 'gaussian' for a Gaussian filter (see gaussian_stddev below)
         - Empty string to use the same filter in the temporal domain as
         the rfilter used in the spatial domain.

         IMPORTANT: RECOMMENDED TO SET TO 'box' FOR NLOS SIMULATIONS. (default: empty string)

     * - block_size
       - |int|
       - Size of (square) image blocks to render in parallel (in scalar mode).
         Should be a power of two. (default: 0 i.e. let Mitsuba decide for you)

     * - max_depth
       - |int|
       - Specifies the longest path depth in the generated output image (where -1
         corresponds to infinity). A value of 1 will only render directly
         visible light sources. 2 will lead to single-bounce (direct-only)
         illumination, and so on. (default: 6)

     * - rr_depth
       - |int|
       - Specifies the path depth, at which the implementation will begin to use
         the *russian roulette* path termination criterion. For example, if set to
         1, then path generation many randomly cease after encountering directly
         visible surfaces. (default: 5)
    """

    def __init__(self, props: mi.Properties):
        super().__init__(props)

        self.filter_depth: int = props.get('filter_depth', -1)
        self.filter_bounces: int = props.get('filter_bounces', -1)
        if self.filter_depth != -1 and self.filter_bounces != -1:
            raise AssertionError(
                'Only use one of filter_depth or filter_bounces')
        if self.filter_bounces != -1:
            self.filter_depth = self.filter_bounces + 1

        if self.filter_depth != -1 and self.max_depth != -1:
            if self.filter_depth >= self.max_depth:
                Log(LogLevel.Warn,
                    'You have set filter_depth >= max_depth. '
                    'This will cause the final image to be all zero.')
        self.discard_direct_paths: bool = props.get(
            'discard_direct_paths', False)
        self.laser_sampling: bool = props.get(
            'nlos_laser_sampling', False)
        self.hg_sampling: bool = props.get(
            'nlos_hidden_geometry_sampling', False)
        self.hg_sampling_do_rroulette = (
            props.get('nlos_hidden_geometry_sampling_do_rroulette', False)
            and
            self.hg_sampling
        )
        self.hg_sampling_includes_relay_wall: bool = (
            props.get('nlos_hidden_geometry_sampling_includes_relay_wall', False)
            and
            self.hg_sampling
        )

    def prepare(self, scene: mi.Scene, sensor: mi.Sensor, seed: mi.UInt32, spp: int, aovs: List):
        # prepare laser sampling
        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        scene_emitters = scene.emitters()
        if len(scene_emitters) > 1:
            Log(LogLevel.Error,
                f'You have defined multiple ({len(scene_emitters)}) emitters in the scene with a NLOS capture meter. You should have only 1.')

        if self.hg_sampling:
            # NOTE (Miguel) : Improved hidden geometry sampling via vectorization.
            # There is a bug that does not allow this currently. Uncomment it after nanobind upgrade
            # ------------------------------------------------------------------------------------
            # valid_object = (scene.shapes_dr() != mi.ShapePtr(sensor.get_shape())) | self.hg_sampling_includes_relay_wall

            # # surface_areas = scene.shapes_dr().surface_area()
            # surface_areas = scene.shapes_dr().surface_area()
            # pdf_hidden_area = dr.gather(mi.Float,
            #                             surface_areas,
            #                             dr.arange(mi.UInt, dr.width(scene.shapes_dr())),
            #                             valid_object)

            # self.hidden_geometries_distribution = mi.DiscreteDistribution(pdf_hidden_area)
            # ------------------------------------------------------------------------------------
            surface_areas = []
            for shape in scene.shapes():
                surface_areas.append(
                    0.0 if (shape == sensor.get_shape()
                            and not self.hg_sampling_includes_relay_wall) else shape.surface_area()[0]
                )

            if len(surface_areas) == 0:
                raise AssertionError('Hidden geometry sampling is activated, '
                                     'but there are no hidden geometries in the scene!')
            if sum(surface_areas) < dr.epsilon(mi.Float):
                raise AssertionError('Hidden geometry sampling is activated, '
                                     'but the hidden geometry in the scene has zero surface area?')

            self.hidden_geometries_distribution = mi.DiscreteDistribution(
                surface_areas)

        # prepare laser sampling by precomputing the laser focusing point in the geometry
        if self.laser_sampling:
            # This integrator expects only one emitter per scene
            trafo: mi.Transform4f = scene.emitters()[0].world_transform()
            laser_origin: mi.Point3f = mi.Point3f(trafo.translation())
            laser_dir: mi.Vector3f = trafo.transform_affine(
                mi.Vector3f(0, 0, 1))

            laser_ray = mi.Ray3f(laser_origin, laser_dir)
            si = scene.ray_intersect(laser_ray)

            assert dr.all(si.is_valid()), \
                'The emitter is not pointing at the scene!'
            self.nlos_laser_target: mi.Point3f = si.p

        return super().prepare(scene, sensor, seed, spp, aovs)

    @dr.syntax
    def _sample_hidden_geometry_position(
            self, ref: mi.Interaction3f, scene: mi.Scene,
            sample2: mi.Point2f, active: mi.Bool) -> mi.PositionSample3f:
        """
        For non-line of sight scenes, sample a point in the hidden
        geometry's surface area. The "hidden geometry" is defined
        as any object that does not contain an \ref nloscapturemeter
        plugin (i.e. every object but the relay wall)

        Parameters
        ----------
        ref
            A reference point somewhere within the scene

        sample2
            A uniformly distributed 2D vector

        active
            A boolean mask

        Returns
        -------
        Position sampling record
        """

        # NOTE(diego): Doing returns inside if's is not supported
        #              if not in scalar mode
        # if len(self.hidden_geometries) == 0:
        #     return dr.zeros(mi.PositionSample3f)

        # if len(self.hidden_geometries) == 1:
        #     return self.hidden_geometries[0].sample_position(
        #         ref.time, sample2, active)

        index, new_sample, shape_pdf = \
            self.hidden_geometries_distribution.sample_reuse_pmf(
                sample2.x, active)
        sample2.x = new_sample

        shape: mi.ShapePtr = dr.gather(
            mi.ShapePtr, scene.shapes_dr(), index, active)
        ps = shape.sample_position(ref.time, sample2, active)
        ps.pdf *= shape_pdf

        return ps

    def emitter_nee_sample(
            self, mode: dr.ADMode, scene: mi.Scene, sampler: mi.Sampler,
            si: mi.SurfaceInteraction3f, bsdf: mi.BSDF, bsdf_ctx: mi.BSDFContext,
            β: mi.Spectrum, distance: mi.Float, η: mi.Float, depth: mi.UInt,
            active_e: mi.Bool, add_transient) -> mi.Spectrum:
        ds, em_weight = scene.sample_emitter_direction(
            ref=si, sample=sampler.next_2d(active_e), test_visibility=True, active=active_e)
        active_e &= (ds.pdf != 0.0)

        primal = mode == dr.ADMode.Primal
        with dr.resume_grad(when=not primal):
            if dr.hint(not primal, mode='scalar'):
                # Given the detached emitter sample, *recompute* its
                # contribution with AD to enable light source optimization
                ds.d = dr.replace_grad(ds.d, dr.normalize(ds.p - si.p))
                em_val = scene.eval_emitter_direction(si, ds, active_e)
                em_weight = dr.replace_grad(em_weight, dr.select(
                    (ds.pdf != 0), em_val / ds.pdf, 0))
                dr.disable_grad(ds.d)

            # Query the BSDF for that emitter-sampled direction
            wo = si.to_local(ds.d)
            bsdf_spec, bsdf_pdf = bsdf.eval_pdf(
                ctx=bsdf_ctx, si=si, wo=wo, active=active_e)

            Lr_dir = mi.Spectrum(0)
            if self.filter_depth != -1:
                active_e &= (depth == self.filter_depth)
            if self.discard_direct_paths:
                active_e &= depth > 2
            Lr_dir[active_e] = β * bsdf_spec * em_weight

        add_transient(Lr_dir, distance + ds.dist * η,
                      si.wavelengths, active_e)

        return Lr_dir

    def emitter_laser_sample(
            self, mode: dr.ADMode, scene: mi.Scene, sampler: mi.Sampler,
            si: mi.SurfaceInteraction3f, bsdf: mi.BSDF, bsdf_ctx: mi.BSDFContext,
            β: mi.Spectrum, distance: mi.Float, η: mi.Float, depth: mi.UInt,
            active_e: mi.Bool, add_transient) -> mi.Spectrum:
        """
        NLOS scenes only have one laser emitter - standard
        emitter sampling techniques do not apply as most
        directions do not emit any radiance, it needs to be very
        lucky to bsdf sample the exact point that the laser is
        illuminating

        this modifies the emitter sampling so instead of directly
        sampling the laser we sample(1) the point that the laser
        is illuminating and then(2) the laser
        """
        # TODO(diego): AD probably needs some "with dr.resume_grad(when=not primal):"
        primal = mode == dr.ADMode.Primal

        # 1. Obtain direction to NLOS illuminated point
        #    and test visibility with ray_test
        d = self.nlos_laser_target - si.p
        distance_laser = dr.norm(d)
        d /= distance_laser
        ray_bsdf = si.spawn_ray_to(self.nlos_laser_target)
        active_e &= ~scene.ray_test(ray_bsdf, active_e)

        # 2. Evaluate BSDF to desired direction
        wo = si.to_local(d)
        bsdf_spec = bsdf.eval(ctx=bsdf_ctx, si=si, wo=wo, active=active_e)
        bsdf_spec = si.to_world_mueller(bsdf_spec, -wo, si.wi)

        ray_bsdf.maxt = dr.inf
        si_bsdf: mi.SurfaceInteraction3f = scene.ray_intersect(
            ray_bsdf, active_e)
        active_e &= si_bsdf.is_valid()
        active_e &= dr.any(mi.depolarizer(bsdf_spec) > dr.epsilon(mi.Float))

        wl = si_bsdf.to_local(-d)
        active_e &= mi.Frame3f.cos_theta(wl) > 0.0
        pdf_ls = mi.Float(1.0)  # always hit the same point :)
        # NOTE(diego): convert from area probability to solid angle probability
        #              (similar to sampling an area light)
        #              divide by pdf
        pdf_ls *= dr.sqr(distance_laser) / mi.Frame3f.cos_theta(wl)
        bsdf_spec /= pdf_ls

        bsdf_next = si_bsdf.bsdf(ray=ray_bsdf)

        # 3. Combine laser + emitter sampling
        return self.emitter_nee_sample(
            mode=mode, scene=scene, sampler=sampler, si=si_bsdf,
            bsdf=bsdf_next, bsdf_ctx=bsdf_ctx, β=β * bsdf_spec, distance=distance + distance_laser * η, η=η,
            depth=depth+1, active_e=active_e, add_transient=add_transient)

    def hidden_geometry_sample(
            self, scene: mi.Scene, sampler: mi.Sampler, bsdf: mi.BSDF, bsdf_ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f,
            _: mi.Float, sample2: mi.Point2f, active: mi.Bool) -> Tuple[mi.BSDFSample3f, mi.Spectrum]:

        if dr.hint(not self.hg_sampling, mode='scalar'):
            return dr.zeros(mi.BSDFSample3f), 0.0

        # repeat HG sampling until we find a visible position (or reach max positions)
        ps_hg: mi.PositionSample3f = self._sample_hidden_geometry_position(
            ref=si, scene=scene, sample2=sample2, active=active)
        d = mi.Vector3f(ps_hg.p - si.p)
        dist = dr.norm(d)
        d /= dist
        cos_theta_i = dr.dot(si.n, d)
        cos_theta_g = dr.dot(ps_hg.n, -d)
        active &= (cos_theta_i > dr.epsilon(mi.Float)) & \
            (cos_theta_g > dr.epsilon(mi.Float))

        wo = si.to_local(d)
        bsdf_spec = bsdf.eval(ctx=bsdf_ctx, si=si, wo=wo, active=active)
        bsdf_spec = si.to_world_mueller(bsdf_spec, -wo, si.wi)

        bs: mi.BSDFSample3f = dr.zeros(mi.BSDFSample3f)
        bs.wo = wo
        bs.pdf = ps_hg.pdf * dr.sqr(dist) / dr.abs(cos_theta_g)
        bs.eta = 1.0
        bs.sampled_type = mi.UInt32(mi.BSDFFlags.Reflection)
        bs.sampled_component = 0
        active &= bs.pdf > dr.epsilon(mi.Float)

        bsdf_spec /= bs.pdf

        return bs, dr.select(active, bsdf_spec, 0.0)

    @dr.syntax
    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               δL: Optional[mi.Spectrum],
               state_in: Optional[mi.Spectrum],
               active: mi.Bool,
               # add_transient accepts (spec, distance, wavelengths, active)
               add_transient: Callable[[mi.Spectrum, mi.Float, mi.UnpolarizedSpectrum, mi.Bool], None],
               **kwargs  # Absorbs unused arguments
               ) -> Tuple[mi.Spectrum, mi.Bool, List[mi.Float], mi.Spectrum]:
        """
        See ``TransientADIntegrator.sample()`` for a description of this interface and
        the role of the various parameters and return values.
        """

        # Rendering a primal image? (vs performing forward/reverse-mode AD)
        primal = mode == dr.ADMode.Primal

        # Standard BSDF evaluation context for path tracing
        bsdf_ctx = mi.BSDFContext()

        # --------------------- Configure loop state ----------------------

        # Copy input arguments to avoid mutating the caller's state
        ray = mi.Ray3f(dr.detach(ray))
        depth = mi.UInt32(0)                          # Depth of current vertex
        L = mi.Spectrum(0 if primal else state_in)    # Radiance accumulator
        # Differential/adjoint radiance
        δL = mi.Spectrum(δL if δL is not None else 0)
        β = mi.Spectrum(1)                            # Path throughput weight
        η = mi.Float(1)                               # Index of refraction
        active = mi.Bool(active)                      # Active SIMD lanes
        distance = mi.Float(ray.time)                 # Distance of the path

        # Variables caching information from the previous bounce
        prev_si = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)

        if self.camera_unwarp:
            raise AssertionError(
                'Do not use camera_unwarp with TransientNLOSPath. Use account_first_and_last_bounces instead for the same purpose.')

        if self.laser_sampling:
            emitter_sample_f = self.emitter_laser_sample
        else:
            emitter_sample_f = self.emitter_nee_sample

        while dr.hint(active,
                      max_iterations=self.max_depth,
                      label="Transient Path (%s)" % mode.name):
            active_next = mi.Bool(active)

            # Compute a surface interaction that tracks derivatives arising
            # from differentiable shape parameters (position, normals, etc.)
            # In primal mode, this is just an ordinary ray tracing operation.
            with dr.resume_grad(when=not primal):
                si = scene.ray_intersect(ray,
                                         ray_flags=mi.RayFlags.All,
                                         coherent=(depth == 0))
            # Update distance
            distance += dr.select(active, si.t, 0.0) * η

            # Get the BSDF, potentially computes texture-space differentials
            bsdf = si.bsdf(ray)

            # ---------------------- Direct emission ----------------------

            # Hide the environment emitter if necessary
            if dr.hint(self.hide_emitters, mode='scalar'):
                active_next &= ~((depth == 0) & ~si.is_valid())

            # Compute MIS weight for emitter sample from previous bounce
            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

            mis = mis_weight(
                prev_bsdf_pdf,
                scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
            )

            with dr.resume_grad(when=not primal):
                Le = β * mis * ds.emitter.eval(si, active_next)

            # Add transient contribution because of emitter found
            add_transient(Le, distance, ray.wavelengths, active)

            # ---------------------- Emitter sampling ----------------------

            # Should we continue tracing to reach one more vertex?
            active_next &= (depth + 1 < self.max_depth) & si.is_valid()

            # Is emitter sampling even possible on the current vertex?
            active_em = active_next & mi.has_flag(
                bsdf.flags(), mi.BSDFFlags.Smooth)

            # Uses NEE or laser sampling depending on self.laser_sampling
            Lr_dir = emitter_sample_f(
                mode, scene, sampler,
                si, bsdf, bsdf_ctx,
                β, distance, η, depth,
                active_em, add_transient)

            # ------------------ Detached BSDF sampling -------------------

            do_hg_sample = mi.Bool(False)
            if self.hg_sampling and self.hg_sampling_do_rroulette:
                # choose HG or BSDF sampling with Russian Roulette
                hg_prob = mi.Float(0.5)
                do_hg_sample = sampler.next_1d(active) < hg_prob
                pdf_bsdf_method = dr.select(
                    do_hg_sample,
                    hg_prob,
                    mi.Float(1.0) - hg_prob)
            else:
                # only one option
                do_hg_sample = mi.Bool(self.hg_sampling)
                pdf_bsdf_method = mi.Float(1.0)

            active_hg = active_next & do_hg_sample
            bsdf_sample_hg, bsdf_weight_hg = self.hidden_geometry_sample(
                scene, sampler, bsdf,
                bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active_hg)

            active_nhg = active_next & (~do_hg_sample)
            bsdf_sample_nhg, bsdf_weight_nhg = bsdf.sample(
                bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active_nhg)

            bsdf_sample = dr.select(
                do_hg_sample, bsdf_sample_hg, bsdf_sample_nhg)
            bsdf_weight = dr.select(
                do_hg_sample, bsdf_weight_hg, bsdf_weight_nhg)

            # ---- Update loop variables based on current interaction -----

            L = (L + Le + Lr_dir) if primal else (L - Le - Lr_dir)
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            η *= bsdf_sample.eta
            β *= bsdf_weight / pdf_bsdf_method

            # Information about the current vertex needed by the next iteration

            prev_si = dr.detach(si, True)
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(
                bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            # -------------------- Stopping criterion ---------------------

            # Don't run another iteration if the throughput has reached zero
            β_max = dr.max(β)
            active_next &= (β_max != 0)

            # Russian roulette stopping probability (must cancel out ior^2
            # to obtain unitless throughput, enforces a minimum probability)
            rr_prob = dr.minimum(β_max * η**2, .95)
            active_next &= rr_prob > 0

            # Apply only further along the path since, this introduces variance
            rr_active = depth >= self.rr_depth
            β[rr_active] *= dr.rcp(rr_prob) & (rr_prob > 0)
            rr_continue = sampler.next_1d() < rr_prob
            active_next &= ~rr_active | rr_continue

            # ------------------ Differential phase only ------------------

            if dr.hint(not primal, mode='scalar'):
                with dr.resume_grad():
                    # 'L' stores the indirectly reflected radiance at the
                    # current vertex but does not track parameter derivatives.
                    # The following addresses this by canceling the detached
                    # BSDF value and replacing it with an equivalent term that
                    # has derivative tracking enabled. (nit picking: the
                    # direct/indirect terminology isn't 100% accurate here,
                    # since there may be a direct component that is weighted
                    # via multiple importance sampling)

                    # Recompute 'wo' to propagate derivatives to cosine term
                    wo = si.to_local(ray.d)

                    # Re-evaluate BSDF * cos(theta) differentiably
                    bsdf_val = bsdf.eval(bsdf_ctx, si, wo, active_next)

                    # Detached version of the above term and inverse
                    bsdf_val_det = bsdf_weight * bsdf_sample.pdf
                    inv_bsdf_val_det = dr.select(bsdf_val_det != 0,
                                                 dr.rcp(bsdf_val_det), 0)

                    # Differentiable version of the reflected indirect
                    # radiance. Minor optional tweak: indicate that the primal
                    # value of the second term is always 1.
                    tmp = inv_bsdf_val_det * bsdf_val
                    tmp_replaced = dr.replace_grad(
                        dr.ones(mi.Float, dr.width(tmp)), tmp)  # FIXME
                    Lr_ind = L * tmp_replaced

                    # Differentiable Monte Carlo estimate of all contributions
                    Lo = Le + Lr_dir + Lr_ind

                    attached_contrib = dr.flag(
                        dr.JitFlag.VCallRecord) and not dr.grad_enabled(Lo)
                    if dr.hint(attached_contrib, mode='scalar'):
                        raise Exception(
                            "The contribution computed by the differential "
                            "rendering phase is not attached to the AD graph! "
                            "Raising an exception since this is usually "
                            "indicative of a bug (for example, you may have "
                            "forgotten to call dr.enable_grad(..) on one of "
                            "the scene parameters, or you may be trying to "
                            "optimize a parameter that does not generate "
                            "derivatives in detached PRB.)")

                    # Propagate derivatives from/to 'Lo' based on 'mode'
                    if dr.hint(mode == dr.ADMode.Backward, mode='scalar'):
                        dr.backward_from(δL * Lo)
                    else:
                        δL += dr.forward_to(Lo)

            depth[si.is_valid()] += 1
            active = active_next

        return (
            L if primal else δL,  # Radiance/differential radiance
            (depth != 0),         # Ray validity flag for alpha blending
            [],                   # Empty typle of AOVs
            L                     # State for the differential phase
        )


mi.register_integrator("transient_nlos_path",
                       lambda props: TransientNLOSPath(props))

del TransientADIntegrator
