# Import/re-import all files in this folder to register AD integrators
import importlib
import mitsuba as mi

if mi.variant() is not None and not mi.variant().startswith('scalar'):
    from . import transient_hdr_film
    importlib.reload(transient_hdr_film)
    from . import phasor_hdr_film
    importlib.reload(phasor_hdr_film)

del importlib, mi
