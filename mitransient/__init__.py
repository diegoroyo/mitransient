# fmt: off
# Check that Mitsuba 3 has a defined variant before executing
try:
    import mitsuba as mi

    if mi.variant() == None:
        raise RuntimeError('''You should define a Mitsuba variant before importing mitransient. For example, you should do:
              
import mitsuba as mi
mi.set_variant('<variant_name>')
import mitransient as mitr
                        
If you installed Mitsuba 3 with pip, you can use llvm_ad_rgb (CPU) or cuda_ad_rgb (GPU) as <variant_name>s.''')

    if mi.variant().startswith('scalar'):
        from mitsuba import Log, LogLevel          
        Log(LogLevel.Warn,
            'You are using a scalar_* variant for Mitsuba. Thus, mitransient will not register many of its plugins. Please switch to a llvm_* or cuda_* variant (e.g. llvm_ad_rgb) if you get messages like "failed to instantiate unknown plugin of type X".')   

except ImportError:
    raise RuntimeError(f'mitransient heavily depends on Mitsuba 3. Please install Mitsuba 3 (pip install mitsuba) before using mitransient.')

# Check that the installed Mitsuba version is compatible
from .version import check_compatibility, __version__
check_compatibility()

# Import all the subpackages of mitransient
from .integrators import *
from .render import *
from .films import *
from .sensors import *
from .emitters import *

from .utils import speed_of_light, cornell_box
from . import nlos
if mi.is_polarized:
    from . import polarized_visualization as vis
else:
    from . import unpolarized_visualization as vis