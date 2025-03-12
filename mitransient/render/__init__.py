# Import/re-import all files in this folder to register AD integrators
import importlib
import mitsuba as mi

if mi.variant() is not None and not mi.variant().startswith('scalar'):
    from . import transient_image_block
    importlib.reload(transient_image_block)
    from . import phasor_image_block
    importlib.reload(phasor_image_block)

del importlib, mi
