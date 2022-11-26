import numpy as np


def draw_3d_pose(keypoints,
                 ax,
                 radius=None,
                 point_size=10,
                 line_width=1,
                 c=None,
                 connectivity=None):
    keypoints_mask = [True] * len(keypoints)

    # Make connection matrix
    for i, joint in enumerate(connectivity):
        if keypoints_mask[joint[0]] and keypoints_mask[joint[1]]:
            xs, ys, zs = [
                np.array([keypoints[joint[0], j], keypoints[joint[1], j]])
                for j in range(3)
            ]

            color = (150, 150, 150)
            color = (np.array(color) / 255).reshape(1, -1)

            ax.plot(xs, ys, zs, lw=line_width, c=color)

    ax.scatter(keypoints[keypoints_mask][:, 0],
               keypoints[keypoints_mask][:, 1],
               keypoints[keypoints_mask][:, 2],
               s=point_size,
               c=c)  # np.array([230, 145, 56]) / 255

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


def fig_to_array(fig):
    fig.canvas.draw()
    fig_image = np.array(fig.canvas.renderer._renderer)

    return fig_image
