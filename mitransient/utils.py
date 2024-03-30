import drjit as dr
import mitsuba as mi
import numpy as np

'''
Constants
'''

speed_of_light = 299792458.0


'''
Define the number of threads to be used
'''


def set_thread_count(count):
    threads_available = mi.util.core_count()
    final_count = max(threads_available, count)
    if count > threads_available:
        mi.Log(mi.LogLevel.Warn,
               f'{threads_available} threads exceeds the number of available cores of your machine ({threads_available}). Setting it to {final_count}.')

    # Main thread counts as one thread
    dr.set_thread_count(final_count - 1)


'''
Define multiple multidimensional arrays
'''


def get_class(name):
    name = name.split('.')
    value = __import__(".".join(name[:-1]))
    for item in name[1:]:
        value = getattr(value, item)
    return value


def get_module(class_):
    return get_class(class_.__module__)

if mi.variant() is None:
    # e.g. pip installation does not set mi.variant()
    # does not matter as it does not use Array variants
    pass
elif mi.variant().startswith('scalar'):
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


'''
Auxiliary functions
'''


def show_video(input_sample, axis_video, uint8_srgb=True):
    # if not in_ipython():
    #     print("[show_video()] needs to be executed in a IPython/Jupyter environment")
    #     return

    import matplotlib.animation as animation
    from IPython.display import HTML, display
    from matplotlib import pyplot as plt
    import numpy as np

    def generate_index(axis_video, dims, index):
        return tuple([np.s_[:] if dim != axis_video else np.s_[index] for dim in range(dims)])

    num_frames = input_sample.shape[axis_video]
    fig = plt.figure()

    frame = input_sample[generate_index(
        axis_video, len(input_sample.shape), 0)]
    im = plt.imshow(mi.util.convert_to_bitmap(frame, uint8_srgb))
    plt.axis('off')

    def update(i):
        frame = input_sample[generate_index(
            axis_video, len(input_sample.shape), i)]
        img = mi.util.convert_to_bitmap(frame, uint8_srgb)
        im.set_data(img)
        return im

    ani = animation.FuncAnimation(fig, update, frames=num_frames, repeat=False)
    display(HTML(ani.to_html5_video()))
    plt.close()


def save_frames(data, axis_video, folder):
    import os
    os.makedirs(folder, exist_ok=True)

    def generate_index(axis_video, dims, index):
        return tuple([np.s_[:] if dim != axis_video else np.s_[index] for dim in range(dims)])

    num_frames = data.shape[axis_video]
    for i in range(num_frames):
        mi.Bitmap(data[generate_index(axis_video, len(data.shape), i)]).write(
            f'{folder}/{i:03d}.exr')

# Indent output of subobjects


def indent(obj, amount=2):
    output = str(obj)
    result = ""
    lines = output.splitlines(keepends=True)
    if len(lines) == 1:
        result += lines[0]
    else:
        for line in lines:
            result += line + ' '*amount
    return result
