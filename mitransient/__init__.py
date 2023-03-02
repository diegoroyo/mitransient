"""
TODO(diego):
- Import mitsuba, if it's not available then raise an exception
    telling the user to source the setpath.sh of the mitsuba installation
    or to install mitsuba using pip.
- Ensure that mitransient has autocompletion with the same attributes as mitsuba
    Needs testing: set the __all__ or __dict__ variables to equal mitsuba's
    __all__ or __dict__ variables, but make sure that our own functions
    overwrite Mitsuba's (e.g. our sensor class)
- Check other files' __init__.py files to see if their contents can be moved
    to the same function
"""


try:
    import mitsuba
except ModuleNotFoundError:
    raise Exception(
        'The mitsuba installation could not be found. '
        'Please install it using pip or source the setpath.sh file of your Mitsuba installation.')

from .integrators import *
from .render import *
from .sensors import *

from .utils import show_video, speed_of_light
