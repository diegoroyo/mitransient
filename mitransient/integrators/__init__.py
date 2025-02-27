# Import/re-import all files in this folder to register AD integrators
import importlib
import mitsuba as mi

if mi.variant() is not None and not mi.variant().startswith('scalar'):
    from . import common
    importlib.reload(common)

    from . import transientpath
    importlib.reload(transientpath)

    from . import transient_prbvolpath
    importlib.reload(transient_prbvolpath)

    from . import transientnlospath
    importlib.reload(transientnlospath)

del importlib, mi
