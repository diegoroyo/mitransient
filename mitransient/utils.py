import drjit as dr
import mitsuba as mi


speed_of_light = 299792458.0
"""Speed of light in meters/second"""


def set_thread_count(count):
    """Define the number of threads to be used"""
    threads_available = mi.misc.core_count()
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


def cornell_box():
    '''
    Returns a dictionary containing a description of the Cornell Box scene for Transient Rendering.
    '''
    T = mi.ScalarTransform4f
    return {
        'type': 'scene',
        'integrator': {
            'type': 'transient_path',
            'camera_unwarp': False,
            'max_depth': 100,
            'temporal_filter': 'box',
            'gaussian_stddev': 2.0,
        },
        # -------------------- Sensor --------------------
        'sensor': {
            'type': 'perspective',
            'fov_axis': 'smaller',
            'near_clip': 0.001,
            'far_clip': 100.0,
            'focus_distance': 1000,
            'fov': 39.3077,
            'to_world': T().look_at(
                origin=[0, 0, 3.90],
                target=[0, 0, 0],
                up=[0, 1, 0]
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': 128
            },
            'film': {
                'type': 'transient_hdr_film',
                'width': 256,
                'height': 256,
                'rfilter': {
                    'type': 'box',
                },
                'temporal_bins': 300,
                'start_opl': 3.5,
                'bin_width_opl': 0.02
            }
        },
        # -------------------- BSDFs --------------------
        'white': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                # 'value': mi.ScalarColor3f([0.885809, 0.698859, 0.666422]),
                'value': mi.ScalarColor3d(0.885809, 0.698859, 0.666422),
            }
        },
        'green': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                # 'value': [0.105421, 0.37798, 0.076425],
                'value': mi.ScalarColor3d(0.105421, 0.37798, 0.076425),
            }
        },
        'red': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                # 'value': [0.570068, 0.0430135, 0.0443706],
                'value': mi.ScalarColor3d(0.570068, 0.0430135, 0.0443706),
            }
        },
        # -------------------- Light --------------------
        'light': {
            'type': 'rectangle',
            'to_world': T().translate([0.0, 0.99, 0.01]).rotate([1, 0, 0], 90).scale([0.23, 0.19, 0.19]),
            'bsdf': {
                'type': 'ref',
                'id': 'white'
            },
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    # 'value': [18.387, 13.9873, 6.75357],
                    'value': mi.ScalarColor3d(18.387, 13.9873, 6.75357),
                }
            }
        },
        # -------------------- Shapes --------------------
        'floor': {
            'type': 'rectangle',
            'to_world': T().translate([0.0, -1.0, 0.0]).rotate([1, 0, 0], -90),
            'bsdf': {
                'type': 'ref',
                'id':  'white'
            }
        },
        'ceiling': {
            'type': 'rectangle',
            'to_world': T().translate([0.0, 1.0, 0.0]).rotate([1, 0, 0], 90),
            'bsdf': {
                'type': 'ref',
                'id':  'white'
            }
        },
        'back': {
            'type': 'rectangle',
            'to_world': T().translate([0.0, 0.0, -1.0]),
            'bsdf': {
                'type': 'ref',
                'id':  'white'
            }
        },
        'green-wall': {
            'type': 'rectangle',
            'to_world': T().translate([1.0, 0.0, 0.0]).rotate([0, 1, 0], -90),
            'bsdf': {
                'type': 'ref',
                'id':  'green'
            }
        },
        'red-wall': {
            'type': 'rectangle',
            'to_world': T().translate([-1.0, 0.0, 0.0]).rotate([0, 1, 0], 90),
            'bsdf': {
                'type': 'ref',
                'id':  'red'
            }
        },
        'small-box': {
            'type': 'cube',
            'to_world': T().translate([0.335, -0.7, 0.38]).rotate([0, 1, 0], -17).scale(0.3),
            'bsdf': {
                'type': 'ref',
                'id':  'white'
            }
        },
        'large-box': {
            'type': 'cube',
            'to_world': T().translate([-0.33, -0.4, -0.28]).rotate([0, 1, 0], 18.25).scale([0.3, 0.61, 0.3]),
            'bsdf': {
                'type': 'ref',
                'id':  'white'
            }
        },
    }
