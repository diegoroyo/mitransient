import mitsuba as mi
import numpy as np


def tonemap_transient(transient, scaling=1.0):
    """Applies a linear tonemapping to the transient image."""
    channel_top = np.quantile(np.array(transient), 0.99)
    return transient / channel_top * scaling


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


def rainbow_visualization(steady_state, data_transient, modulo, min_modulo, max_modulo, max_time_bins=None, mode="peak_time_fusion", scale_full_fusion=1):
    import matplotlib.cm as cm
    time_bins = data_transient.shape[2] if max_time_bins is None else max_time_bins

    # From: http://giga.cps.unizar.es/~ajarabo/pubs/MT/downloads/Jarabo2012_MasterThesis_noannex.pdf
    # 1. Obtain full fusion (a.k.a steady rendering)
    full_fusion = np.sum(data_transient, axis=2)

    # 2. Obtain idx of time bin with peak radiance for every pixel i,j
    # 2.1 Peak radiance according to the max value of RGB for every pixel and time_bin
    data_transient_max = np.max(data_transient, axis=-1)
    # 2.2 Compute index
    idx_data_transient_max = np.argmax(data_transient_max, axis=-1)

    # 3. Create visualization
    visualization = np.zeros(data_transient.shape)
    for i in range(visualization.shape[0]):
        for j in range(visualization.shape[1]):
            idx = idx_data_transient_max[i, j]

            # Sparse fusion
            if mode == "sparse_fusion":
                if min_modulo <= idx % modulo <= max_modulo:
                    visualization[i,j,idx,:] = data_transient[i,j,idx,:]**scale_full_fusion
                else:
                    visualization[i,j,idx,:] = 0
            elif mode == "rainbow_fusion":
                if min_modulo <= idx % modulo <= max_modulo:
                    visualization[i,j,idx,:] = cm.jet(idx/time_bins)[:3]
                else:
                    visualization[i,j,idx,:] = 0
            elif mode == "peak_time_fusion":
                if min_modulo <= idx % modulo <= max_modulo:
                    visualization[i,j,idx,:] = cm.jet(idx/time_bins)[:3]
                else:
                    visualization[i,j,idx,:] = steady_state[i,j,:]
            else:
                raise NotImplementedError("Not implemented")
    
    visualization = np.sum(visualization, axis=2)
    return visualization