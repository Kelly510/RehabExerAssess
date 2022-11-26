import os
import numpy as np
from scipy.spatial.transform import Rotation as R


def construct_skeleton_kinect(position, angle):
    '''
    Construct human skeleton from Kinect data.
    Position and angle are given relatively to each joint's father joint, except the root joint. 
    '''
    HUMAN_TREE = {
        0: [1, 14, 18],
        1: [2],
        14: [15],
        18: [19],
        2: [3],
        15: [16],
        19: [20],
        3: [4, 6, 10],
        16: [17],
        20: [21],
        4: [5],
        6: [7],
        10: [11],
        7: [8],
        11: [12],
        8: [9],
        12: [13]
    }
    HUMAN_TREE_LAYER = [[0], [1, 14, 18], [2, 15, 19], [3, 16, 20], [4, 6, 10],
                        [7, 11], [8, 12]]

    abs_position = np.zeros((position.shape[0], 3))
    abs_position[0] = position[0]
    abs_angle = np.zeros((angle.shape[0], 3, 3))

    abs_angle[0] = np.array(
        R.from_euler('xyz', angle[0], degrees=True).as_matrix())
    for layer in HUMAN_TREE_LAYER:
        for b in layer:
            for e in HUMAN_TREE[b]:
                abs_angle[e] = np.array(
                    R.from_euler('xyz', angle[e], degrees=True).as_matrix())
                abs_angle[e] = np.matmul(abs_angle[e], abs_angle[b])

    for layer in HUMAN_TREE_LAYER:
        for b in layer:
            for e in HUMAN_TREE[b]:
                abs_position[e] = np.matmul(abs_angle[b],
                                            position[e]) + abs_position[b]

    rotmat = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    abs_position = np.matmul(abs_position, rotmat)
    return abs_position


def construct_skeleton_vicon(position, angle):
    '''
    Construct human skeleton from Vicon data.
    Position data itself is skeletons. 
    '''
    rotmat = np.array([[0, -1, 0], (1, 0, 0), (0, 0, 1)])
    position = np.matmul(position, rotmat)
    position = position / 10
    return position


def load_data(c, k, m, s, e):
    prefix = 'Correct' if c else 'Incorrect'
    device = 'Kinect' if k else 'Vicon'
    construct_function = construct_skeleton_kinect if k else construct_skeleton_vicon
    if c:
        ang_path = os.path.join(
            prefix, device, 'Angles',
            'm{:02d}_s{:02d}_e{:02d}_angles.txt'.format(m, s, e))
    else:
        ang_path = os.path.join(
            prefix, device, 'Angles',
            'm{:02d}_s{:02d}_e{:02d}_angles_inc.txt'.format(m, s, e))
    angles = np.loadtxt(ang_path, delimiter=',')
    angles = angles.reshape(angles.shape[0], -1, 3)
    if c:
        pos_path = os.path.join(
            prefix, device, 'Positions',
            'm{:02d}_s{:02d}_e{:02d}_positions.txt'.format(m, s, e))
    else:
        pos_path = os.path.join(
            prefix, device, 'Positions',
            'm{:02d}_s{:02d}_e{:02d}_positions_inc.txt'.format(m, s, e))
    positions = np.loadtxt(pos_path, delimiter=',')
    positions = positions.reshape(positions.shape[0], -1, 3)

    keypoints = []
    for frame_idx in range(positions.shape[0]):
        angle = angles[frame_idx]
        position = positions[frame_idx]
        keypoints.append(construct_function(position, angle))

    return np.array(keypoints)


if __name__ == '__main__':
    c = False
    k = False
    root_path = os.path.join('Correct' if c else 'Incorrect',
                             'Kinect' if k else 'Vicon', 'Angles')
    save_path = os.path.join('Correct' if c else 'Incorrect',
                             'Kinect' if k else 'Vicon', 'Skeletons')
    for i, filename in enumerate(os.listdir(root_path)):
        fileinfo = filename.split('_')
        m = int(fileinfo[0][1:])
        s = int(fileinfo[1][1:])
        e = int(fileinfo[2][1:])
        keypoints = load_data(c, k, m, s, e)
        keypoints = keypoints.reshape(keypoints.shape[0], -1)
        np.savetxt(os.path.join(save_path,
                                '_'.join(filename.split('_')[:3]) + '.txt'),
                   keypoints,
                   fmt='%.04f')
