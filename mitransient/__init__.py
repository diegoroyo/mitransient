# fmt: off
# Check that Mitsuba 3 has a defined variant before executing
try:
    import mitsuba as mi

    if mi.variant() == None:
        raise RuntimeError('''You should define a Mitsuba variant before importing `mitransient`. For example, you should do:
                           
import mitsuba as mi
mi.set_variant('<variant_name>')
import mitransient as mitr
                        
If you installed Mitsuba 3 with pip, you can use llvm_ad_rgb (CPU) or cuda_ad_rgb (GPU) as <variant_name>s.''')

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

from .utils import speed_of_light, cornell_box
from . import nlos
from . import visualization as vis