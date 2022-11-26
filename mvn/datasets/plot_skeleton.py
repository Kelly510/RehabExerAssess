import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def draw_3d_pose(keypoints,
                 ax,
                 connectivity,
                 radius=None,
                 point_size=10,
                 line_width=1,
                 c=None):

    for i, joint in enumerate(connectivity):
        xs, ys, zs = [
            np.array([keypoints[joint[0], j], keypoints[joint[1], j]])
            for j in range(3)
        ]

        color = (150, 150, 150)
        color = (np.array(color) / 255).reshape(3, )

        ax.plot(xs, ys, zs, lw=line_width, c=color)

    ax.scatter(keypoints[:, 0],
               keypoints[:, 1],
               keypoints[:, 2],
               s=point_size,
               c=c)

    if radius is not None:
        root = np.mean(keypoints, axis=0)
        xroot, yroot, zroot = root
        ax.set_xlim([-radius + xroot, radius + xroot])
        ax.set_ylim([-radius + yroot, radius + yroot])
        ax.set_zlim([-radius + zroot, radius + zroot])

    # Get rid of the panes
    background_color = np.array([252, 252, 252]) / 255

    ax.w_xaxis.set_pane_color(background_color)
    ax.w_yaxis.set_pane_color(background_color)
    ax.w_zaxis.set_pane_color(background_color)

    # Get rid of the ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Mark the axes' name
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def fig_to_array(fig):
    fig.canvas.draw()
    fig_image = np.array(fig.canvas.renderer._renderer)

    return fig_image


def plot_skeleton(keypoints, connectivity, savepath, radius=1, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(savepath, fourcc, fps, (640, 480), True)
    for frame_idx in range(keypoints.shape[0]):
        fig = plt.figure()
        ax = Axes3D(fig)
        draw_3d_pose(keypoints[frame_idx],
                     ax,
                     connectivity,
                     radius=radius,
                     c=[[0, 0, 0]] * keypoints.shape[1])
        img = np.ascontiguousarray(fig_to_array(fig))[..., (2, 1, 0)]
        writer.write(img)
        plt.clf()
        plt.close('all')
    writer.release()
    print('Video saved in {}'.format(savepath))
