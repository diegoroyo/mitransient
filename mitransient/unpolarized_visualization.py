"""
    General visualization utilities for transient rendering.

    .. warning::
        This module is accesible through ``mitransient.vis``, NOT ``mitransient.unpolarized_visualization``.
        The functions are accessible through ``mitransient.vis`` when Mitsuba is NOT running with a ``*_polarized`` variant.
        Else it uses the functions from ``mitransient.polarized_visualization``.
"""

import mitsuba as mi
import numpy as np


def tonemap_transient(transient, scaling=1.0):
    """Applies a linear tonemapping to the transient image."""
    tnp = np.array(transient)
    channel_top = np.quantile(np.abs(tnp), 0.99)
    return tnp / channel_top * scaling


def tonemap_grad_transient(transient, axis_video=2):
    """Converts a gradient video to the coolwarm colormap."""
    assert axis_video == 2
    tnp = np.array(transient)
    if tnp.ndim == 4:
        tnp = np.mean(tnp, axis=-1)
    max_val = np.quantile(np.abs(tnp), 0.999)
    tnp_tonemapped = np.zeros((*tnp.shape[0:3], 3), dtype=np.float32)
    tnp_tonemapped[..., 0] = tnp
    tnp_tonemapped /= max_val
    tnp_tonemapped = np.clip(tnp_tonemapped, -1, 1)
    nt = tnp.shape[axis_video]
    from matplotlib import cm
    colormap = cm.get_cmap('coolwarm')
    for i in range(nt):
        frame = tnp_tonemapped[:, :, i, 0]
        frame_norm = (frame + 1) / 2
        tnp_tonemapped[:, :, i, :] = colormap(frame_norm)[:, :, :3]
    return tnp_tonemapped


def save_video(path, transient, axis_video=2, fps=24, display_video=False):
    """Saves the transient image in video format (.mp4)."""
    import cv2

    def generate_index(axis_video, dims, index):
        return tuple([np.s_[:] if dim != axis_video else np.s_[index] for dim in range(dims)])

    size = (transient.shape[1], transient.shape[0])
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(transient.shape[axis_video]):
        frame = transient[generate_index(axis_video, len(transient.shape), i)]
        bitmap = mi.Bitmap(frame).convert(
            component_format=mi.Struct.Type.UInt8, srgb_gamma=True)
        out.write(np.array(bitmap)[:, :, ::-1])

    out.release()

    if display_video:
        from IPython.display import Video, display
        return display(Video(path, embed=True, width=size[0], height=size[1]))


def save_frames(data, folder, axis_video=2):
    """Saves the transient image in separate frames (.exr format for each frame)."""
    import os
    os.makedirs(folder, exist_ok=True)

    def generate_index(axis_video, dims, index):
        return tuple([np.s_[:] if dim != axis_video else np.s_[index] for dim in range(dims)])

    num_frames = data.shape[axis_video]
    for i in range(num_frames):
        mi.Bitmap(data[generate_index(axis_video, len(data.shape), i)]).write(
            f'{folder}/{i:03d}.exr')


def show_video(input_sample, axis_video=2, uint8_srgb=True):
    """
    Shows the transient video in a IPython/Jupyter environment.

    :param input_sample: array representing the transient image
    :param int axis_video: axis of the array for the temporal dimension
    :param bool uint8_srgb: precision to use when converting to bitmap each frame of the video
    """
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


def rainbow_visualization(steady_state, data_transient,
                          modulo, min_modulo, max_modulo,
                          max_time_bins=None, mode='peak_time_fusion',
                          scale_fusion=1):
    import matplotlib.cm as cm
    # From: http://giga.cps.unizar.es/~ajarabo/pubs/MT/downloads/Jarabo2012_MasterThesis_noannex.pdf
    time_bins = data_transient.shape[2] if max_time_bins is None else max_time_bins

    # Compute the time bin index with the peak radiance for each pixel
    idx = np.argmax(np.max(data_transient, axis=-1), axis=-1)

    valid = (idx % modulo >= min_modulo) & (idx % modulo <= max_modulo)

    # Rainbow colors
    colors = cm.jet(idx / time_bins)[..., :3]

    # Output image: one color per pixel (H, W, channels)
    result = np.zeros_like(steady_state)
    if mode == "sparse_fusion":
        # Select the data at the peak index for each pixel
        result[valid] = steady_state[valid]**scale_fusion
    elif mode == "rainbow_fusion":
        result[valid] = colors[valid]
    elif mode == "peak_time_fusion":
        result[valid] = colors[valid]
        result[~valid] = steady_state[~valid]**scale_fusion
    else:
        raise NotImplementedError("Mode not implemented")

    return result
