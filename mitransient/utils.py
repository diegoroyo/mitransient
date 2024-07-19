import drjit as dr
import mitsuba as mi


speed_of_light = 299792458.0
"""Speed of light in meters/second"""


def set_thread_count(count):
    """Define the number of threads to be used"""
    threads_available = mi.util.core_count()
    final_count = max(threads_available, count)
    if count > threads_available:
        mi.Log(mi.LogLevel.Warn,
               f'{threads_available} threads exceeds the number of available cores of your machine ({threads_available}). Setting it to {final_count}.')

    # Main thread counts as one thread
    dr.set_thread_count(final_count - 1)


def indent(obj, amount=2):
    """Indent output of subobjects"""
    output = str(obj)
    result = ""
    lines = output.splitlines(keepends=True)
    if len(lines) == 1:
        result += lines[0]
    else:
        for line in lines:
            result += line + ' '*amount
    return result


# Define multiple multidimensional arrays
def get_class(name):
    name = name.split('.')
    value = __import__(".".join(name[:-1]))
    for item in name[1:]:
        value = getattr(value, item)
    return value


def get_module(class_):
    return get_class(class_.__module__)


if mi.variant().startswith('scalar'):
    ArrayXf = dr.scalar.ArrayXf
    ArrayXu = dr.scalar.ArrayXu
    ArrayXi = dr.scalar.ArrayXi
    Array2f = dr.scalar.Array2f
    Array2u = dr.scalar.Array2u
    Array3f = dr.scalar.Array3f
else:
    ArrayXf = get_module(mi.Float).ArrayXf
    ArrayXu = get_module(mi.Float).ArrayXu
    ArrayXi = get_module(mi.Float).ArrayXi
    Array2f = get_module(mi.Float).Array2f
    Array2u = get_module(mi.Float).Array2u
    Array3f = get_module(mi.Float).Array3f
