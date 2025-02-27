# Check that Mitsuba 3 has a defined variant before executing
try:
from . import visualization as vis
from . import nlos
from .utils import speed_of_light, cornell_box
from .sensors import *
from .films import *
from .render import *
from .integrators import *
import mitsuba as mi

if mi.variant() == None:
    raise RuntimeError('''You should define a Mitsuba variant before importing `mitransient`. For example, you should do:
    import mitsuba as mi
    mi.set_variant('<variant_name>')
    import mitransient as mitr''')

except ImportError:
    raise RuntimeError(
        f'This library heavily depends on Mitsuba 3. Please install it before using it.')

# Check that the installed Mitsuba version is compatible
from .version import check_compatibility, __version__
check_compatibility()

# Import all the subpackages of mitransient
