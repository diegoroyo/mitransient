# Import/re-import all files in this folder to register AD integrators
import importlib
import mitsuba as mi

if mi.variant() is not None and not mi.variant().startswith('scalar'):
    from . import nlossensor
    importlib.reload(nlossensor)

    from . import nloscapturemeter
    importlib.reload(nloscapturemeter)
del importlib, mi
