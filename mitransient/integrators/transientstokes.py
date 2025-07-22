from __future__ import annotations  # Delayed parsing of type annotations
from typing import Optional, Tuple, List, Callable

import drjit as dr
import mitsuba as mi
from mitsuba.ad.integrators.common import mis_weight  # type: ignore

from .common import TransientADIntegrator


class TransientStokes(TransientADIntegrator):
    r"""
    .. _integrator-transient_stokes:

    Transient Path (:monosp:`transient_stokes`)
    -----------------------------------------

    Standard path tracing algorithm which now includes the time dimension.
    This can render line-of-sight (LOS) scenes. The `transient_nlos_path` 
    plugin contains different sampling routines specific to NLOS setups.
    Choose one or the other depending on if you have a LOS or NLOS scene.
    """
    def __init__(self, props: mi.Properties):
        super().__init__(props)  # initialize props: max_depth and rr_depth
        self.integrator = props.get('integrator')

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
               add_transient: Callable[[mi.Spectrum, mi.Float, mi.UnpolarizedSpectrum, mi.Mask], None],
               **kwargs  # Absorbs unused arguments
               ) -> Tuple[mi.Spectrum, mi.Bool, List[mi.Float], mi.Spectrum]:
        L, valid, aovs, extra = self.integrator.sample(
            mode=mode,
            scene=scene,
            sampler=sampler,
            ray=ray,
            δL=δL,
            state_in=state_in,
            active=active,
            add_transient=add_transient,
            **kwargs
        )
        return L, valid, aovs, extra


mi.register_integrator("transient_stokes", lambda props: TransientStokes(props))

del TransientADIntegrator