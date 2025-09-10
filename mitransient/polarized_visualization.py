"""
    Polarized version of "mitransient/unpolarized_visualization.py"

    .. warning::
        This module is accesible through ``mitransient.vis``, NOT ``mitransient.polarized_visualization``.
        The functions are accessible through ``mitransient.vis`` when Mitsuba is running with a ``*_polarized`` variant.
        Else it uses the functions from ``mitransient.unpolarized_visualization``.
"""
from enum import Enum
import mitsuba as mi
import numpy as np
import cv2
import os


def plot_to_numpy(fig):
    """
    Convert matplotlib figure into a rasterized image, stored as a numpy array.

    :param fig: Matplotlib figure to convert
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()

    return np.moveaxis(np.asarray(buf)[:, :, :-1], [0, 1, 2], [1, 2, 0])


DisplayMethod = Enum(
    'DisplayMethod', [('ShowVideo', 1), ('SaveVideo', 2), ('SaveFrames', 3)])


def show_video_polarized(stokes, degree_of_polarization, angle_of_polarization, type_of_polarization, chirality, save_path=None,
                         display_method=DisplayMethod.ShowVideo, fps=24, intensity_cutoff=0.9, gamma=1.1, show_false_color=True):
    """
    Displays or saves the transient-polarized video.

    :param stokes: transient stokes vector array
    :param degree_of_polarization: false color degree of polarization 
    :param angle_of_polarization: false color angle or linear polarization
    :param type_of_polarization: false color type of polarization
    :param chirality: false color chirality of circular polarization
    :param save_path: name of the folder where the transient frames will be saved / name of file of the transient video
    :param display_method: ShowVideo shows the video in a Jupyter environment, SaveVideo saves it to a .mp4 file, SaveFrames saves it frame by frame
    :param fps: framerate of the video to be displayed/saved
    :param intensity_cutoff: intenisty quantile set as the maximum value during the tonemapping
    :param gamma: gamma value for the tonemapping of the intensity component
    :param show_false_color: if True shows the false color visualization, hidden otherwise
    """
    import logging
    import matplotlib.pyplot as plt
    logging.getLogger('matplotlib').setLevel(logging.ERROR)  # Supress matplotlib warnings

    stokes_np = np.array(stokes)
    timebins = stokes.shape[-2]

    # Normalize Stokes vector with the intensity component
    stokes_np[..., 1:] = stokes_np[..., 1:] / \
        np.maximum(stokes_np[..., [0]], 0.0001)
    stokes12_max = np.maximum(np.abs(
        np.max(stokes_np[..., 1:3], axis=None)), np.abs(np.min(stokes_np[..., 1:3])))
    stokes3_max = np.maximum(
        np.abs(np.max(stokes_np[..., 3], axis=None)), np.abs(np.min(stokes_np[..., 3])))

    # Intensity component threshold and gamma tonemapping
    stokes0_max = np.quantile(stokes_np[..., 0], intensity_cutoff, axis=None)
    stokes_np[..., 0] = np.power(stokes_np[..., 0], 1.0 / gamma)

    # Setup subplots
    nrows = 2 if show_false_color else 1
    fig, axes = plt.subplots(nrows=nrows, ncols=4, figsize=(10, 6))
    plt.subplots_adjust(wspace=0.22, hspace=0.0)
    for ax in axes.ravel():
        ax.axis('off')

    # Plot a frame of the visualization
    def plot_polarimetric_frame(timebin, stokes, degree_of_polarization, angle_of_polarization, type_of_polarization, chirality, show_false_color):
        images = list()
        # Plot Stokes vector
        plt.subplot(nrows, 4, 1)
        images.append(plt.imshow(
            stokes_np[:, :, timebin, 0], cmap='Greys_r', vmin=0.0, vmax=stokes0_max))
        plt.title('s0')
        plt.subplot(nrows, 4, 2)
        images.append(plt.imshow(
            stokes_np[:, :, timebin, 1], cmap='seismic', vmin=-stokes12_max, vmax=stokes12_max))
        plt.title('s1')
        plt.subplot(nrows, 4, 3)
        images.append(plt.imshow(
            stokes_np[:, :, timebin, 2], cmap='seismic', vmin=-stokes12_max, vmax=stokes12_max))
        cbar_stokes12 = plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('s2')
        plt.axis('off')
        plt.subplot(nrows, 4, 4)
        images.append(plt.imshow(
            stokes_np[:, :, timebin, 3], cmap='seismic', vmin=-stokes3_max, vmax=stokes3_max))
        cbar_stokes3 = plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('s3')
        plt.axis('off')

        cbar_stokes12.ax.tick_params(labelsize=7)
        cbar_stokes3.ax.tick_params(labelsize=7)

        # Plot false color images
        if show_false_color:
            plt.subplot(nrows, 4, 5)
            images.append(plt.imshow(degree_of_polarization[:, :, timebin, :]))
            plt.title('Degree of polarization')
            plt.subplot(nrows, 4, 6)
            images.append(plt.imshow(angle_of_polarization[:, :, timebin, :]))
            plt.title('Angle of polarization')
            plt.subplot(nrows, 4, 7)
            images.append(plt.imshow(type_of_polarization[:, :, timebin, :]))
            plt.title('Type of polarization')
            plt.subplot(nrows, 4, 8)
            images.append(plt.imshow(chirality[:, :, timebin, :]))
            plt.title('Chirality')

        return images

    # Update the visualitzation figure with data from a new frame
    def plot_polarimetric_frame_update(timebin, images, stokes, degree_of_polarization, angle_of_polarization, type_of_polarization, chirality, show_false_color):
        # Plot Stokes vector
        images[0].set_data(stokes_np[:, :, timebin, 0])
        images[1].set_data(stokes_np[:, :, timebin, 1])
        images[2].set_data(stokes_np[:, :, timebin, 2])
        images[3].set_data(stokes_np[:, :, timebin, 3])

        # Plot false color images
        if show_false_color:
            images[4].set_data(degree_of_polarization[:, :, timebin, :])
            images[5].set_data(angle_of_polarization[:, :, timebin, :])
            images[6].set_data(type_of_polarization[:, :, timebin, :])
            images[7].set_data(chirality[:, :, timebin, :])
        return images

    # Show video in Jupyter notebook
    if display_method == DisplayMethod.ShowVideo:
        import matplotlib.animation as animation
        from IPython.display import HTML, display
        from functools import partial

        images = plot_polarimetric_frame(
            0, stokes, degree_of_polarization, angle_of_polarization, type_of_polarization, chirality, show_false_color)
        ani = animation.FuncAnimation(fig,
                                      partial(plot_polarimetric_frame_update, images=images, stokes=stokes, degree_of_polarization=degree_of_polarization, angle_of_polarization=angle_of_polarization,
                                              type_of_polarization=type_of_polarization, chirality=chirality, show_false_color=show_false_color),
                                      frames=range(timebins), interval=1 / fps * 1e3, blit=True, repeat=False)
        display(HTML(ani.to_html5_video()))
        plt.close()

    # Save video to file
    elif display_method == DisplayMethod.SaveVideo:
        # Initial figure
        images = plot_polarimetric_frame(
            0, stokes, degree_of_polarization, angle_of_polarization, type_of_polarization, chirality, show_false_color)

        # Create video writer
        frame_init = plot_to_numpy(fig)
        height = frame_init.shape[1]
        width = frame_init.shape[2]
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(
            *'mp4v'), fps, (width, height))

        # Write each frame and save as video
        for timebin in range(timebins):
            plot_polarimetric_frame_update(timebin, images, stokes, degree_of_polarization,
                                           angle_of_polarization, type_of_polarization, chirality, show_false_color)
            frame = plot_to_numpy(fig)
            frame = np.swapaxes(np.swapaxes(frame, 0, -1), 0, 1)

            bitmap = mi.Bitmap(frame).convert(
                component_format=mi.Struct.Type.UInt8, srgb_gamma=True)
            out.write(np.array(bitmap)[:, :, ::-1])

        out.release()

    # Save frames to separate files
    else:  # (DisplayMethod.SaveFrames)
        os.makedirs(save_path, exist_ok=True)
        images = plot_polarimetric_frame(
            0, stokes, degree_of_polarization, angle_of_polarization, type_of_polarization, chirality, show_false_color)

        for timebin in range(timebins):
            plot_polarimetric_frame_update(timebin, images, stokes, degree_of_polarization,
                                           angle_of_polarization, type_of_polarization, chirality, show_false_color)
            plt.savefig(f'{save_path}/{timebin:03d}.png',
                        bbox_inches='tight', pad_inches=0)


def degree_of_polarization(stokes, s0_minimum=0.01):
    """
    Computes the degree of polarization of an array of Stokes vectors.

    :param stokes: Input array of Stokes vectors, with the Stokes components stored as the trailing axis of the array. 
    :param s0_minimum: Minimum value of the s0 component, to avoid NaNs caused by dividing by 0. Default value of 0.1.
    """
    dop = np.sqrt(np.square(stokes[..., 1]) + np.square(stokes[..., 2]) +
                  np.square(stokes[..., 3])) / np.maximum(stokes[..., 0], s0_minimum)
    dop = np.clip(dop, a_min=0.0, a_max=1.0)
    return dop


def degree_of_linear_polarization(stokes, s0_minimum=0.01):
    """
    Computes the degree of linear polarization of an array of Stokes vectors.

    :param stokes: Input array of Stokes vectors, with the Stokes components stored as the trailing axis of the array. 
    :param s0_minimum: Minimum value of the s0 component, to avoid NaNs caused by dividing by 0. Default value of 0.1.
    """
    dop = np.sqrt(np.square(
        stokes[..., 1]) + np.square(stokes[..., 2])) / np.maximum(stokes[..., 0], s0_minimum)
    dop = np.clip(dop, a_min=0.0, a_max=1.0)
    return dop


def degree_of_circular_polarization(stokes, s0_minimum=0.01):
    """
    Computes the degree of circular polarization of an array of Stokes vectors.

    :param stokes: Input array of Stokes vectors, with the Stokes components stored as the trailing axis of the array. 
    :param s0_minimum: Minimum value of the s0 component, to avoid NaNs caused by dividing by 0. Default value of 0.1.
    """
    dop = np.sqrt(np.square(stokes[..., 3])) / \
        np.maximum(stokes[..., 0], s0_minimum)
    dop = np.clip(dop, a_min=0.0, a_max=1.0)
    return dop


def polarization_generate_false_color(stokes, aolp_intensity_scaling=1.0, s0_minimum=0.01):
    """
    Given an input array of Stokes vectors, computes the false color visualizations of the degree of polarization, 
    angle of polarization, type of polarization and chirality of circular polarization, following the format 
    proposed in 'A standardised polarisation visualisation for images' [Wilkie and Weidlich, 2010].

    :param stokes: Input array of Stokes vectors, with the Stokes components stored as the trailing axis of the array. 
    :param aolp_intensity_scaling: Scale value for plotting the angle of linear polarization. Default value of 1.
    :param s0_minimum: Minimum value of the s0 component, to avoid NaNs caused by dividing by 0. Default value of 0.1. 
    """
    def clone_tensor(x, times):
        return np.repeat(x[..., np.newaxis], times, axis=-1)

    # Compute the degree of polarization (total, linear and circular) of the input Stokes vectors
    dop = degree_of_polarization(stokes, s0_minimum)
    dop_linear = degree_of_linear_polarization(stokes, s0_minimum)
    dop_circular = degree_of_circular_polarization(stokes, s0_minimum)

    # False degree of polarization color: low values map to black, high values map to red
    dop_reds = clone_tensor(dop, 3) * np.array([255.0, 0.0, 0.0])

    # Angle of linear polarization, rainbow colormap represents the orientation of the oscillation plane
    # Generate colors from the values of the linear polarization Stokes components
    s1_norm = stokes[..., 1] / np.maximum(stokes[..., 0], s0_minimum)
    s2_norm = stokes[..., 2] / np.maximum(stokes[..., 0], s0_minimum)
    s1_color = np.stack([
        np.maximum(-s1_norm, 0.0),
        np.maximum(s1_norm, 0.0),
        np.zeros((stokes[..., 0].shape))
    ], axis=-1)
    s2_color = np.stack([
        np.maximum(s2_norm, 0.0),
        np.maximum(s2_norm, 0.0),
        np.maximum(-s2_norm, 0.0)
    ], axis=-1)

    # Combine Stokes 1 & 2, add scaling with the linear degree of polarization
    aolp_rainbow = (s1_color + s2_color) * 255.0
    aolp_rainbow_scaled = aolp_rainbow * \
        aolp_intensity_scaling * clone_tensor(dop_linear, 3)

    # Type of polarization: blue for linear, yellow for circular
    linear_polarization_color = np.array([22.0, 247.0, 247.0])
    circular_polarization_color = np.array([252.0, 251.0, 10.0])
    type_of_polarization = clone_tensor(
        dop_linear, 3) * linear_polarization_color + clone_tensor(dop_circular, 3) * circular_polarization_color

    # Chirality: blue for right circular orientation, yellow for left circular orientation
    s3_norm = stokes[..., 3] / np.maximum(stokes[..., 0], s0_minimum)
    chirality = np.stack([
        np.maximum(s3_norm, 0.0),
        np.maximum(s3_norm, 0.0),
        np.maximum(-s3_norm, 0.0)
    ], axis=-1)
    chirality *= clone_tensor(dop_circular, 3)
    chirality *= 255.0

    return dop_reds, aolp_rainbow, aolp_rainbow_scaled, type_of_polarization, chirality


def tonemap_transient(transient, scaling=1.0, normalize_M00=False):
    """Applies a linear tonemapping to the transient image."""
    tnp = np.array(transient)
    channel_top = np.quantile(np.abs(tnp), 0.99)

    # Polarized case
    if normalize_M00:
        tnp[..., 1:] = tnp[..., 1:] / tnp[..., [0]]

    tnp[np.isnan(tnp)] = 0

    return tnp / channel_top * scaling


def save_video(path, transient, axis_video, fps=24, display_video=False):
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


def save_frames(data, axis_video, folder):
    """Saves the transient image in separate frames (.exr format for each frame)."""
    import os
    os.makedirs(folder, exist_ok=True)

    def generate_index(axis_video, dims, index):
        return tuple([np.s_[:] if dim != axis_video else np.s_[index] for dim in range(dims)])

    num_frames = data.shape[axis_video]
    for i in range(num_frames):
        mi.Bitmap(data[generate_index(axis_video, len(data.shape), i)]).write(
            f'{folder}/{i:03d}.exr')


def show_video(input_sample, axis_video, uint8_srgb=True):
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
