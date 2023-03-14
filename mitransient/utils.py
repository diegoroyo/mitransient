import drjit as dr
import mitsuba as mi

'''
Constants
'''

speed_of_light = 299792458.0


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


ArrayXf = get_module(mi.Float).ArrayXf
ArrayXu = get_module(mi.Float).ArrayXu
ArrayXi = get_module(mi.Float).ArrayXi
Array2f = get_module(mi.Float).Array2f
Array2u = get_module(mi.Float).Array2u
Array3f = get_module(mi.Float).Array3f


'''
Auxiliary functions
'''


def show_video(input_sample, axisVideo):
    # if not in_ipython():
    #     print("[showVideo()] Need to be executed in IPython/Jupyter environment")
    #     return

    import matplotlib.animation as animation
    from IPython.display import HTML, display
    from matplotlib import pyplot as plt
    import numpy as np

    def generateIndex(axisVideo, dims, index):
        return tuple([np.s_[:] if dim != axisVideo else np.s_[index] for dim in range(dims)])

    numFrames = input_sample.shape[axisVideo]
    fig = plt.figure()

    im = plt.imshow(input_sample[generateIndex(
        axisVideo, len(input_sample.shape), 0)])
    plt.axis('off')

    def update(i):
        img = input_sample[generateIndex(
            axisVideo, len(input_sample.shape), i)]
        im.set_data(img)
        return im

    ani = animation.FuncAnimation(fig, update, frames=numFrames, repeat=False)
    # display(HTML(ani.to_html5_video()))
    display(HTML(ani.to_html5_video()))
