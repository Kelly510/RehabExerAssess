import os
import numpy as np
from torch.utils.data import Dataset


class UIPRMD_Kinect_Dataset(Dataset):

    def __init__(self, dataset_root=None, movement='m01', is_train=True):
        super().__init__()
        self.dataset_root = os.path.join(dataset_root, 'data', 'UI-PRMD')
        self.is_train = is_train
        self.connectivity = [
            (5, 4),
            (4, 3),
            (3, 2),
            (2, 1),
            (1, 0),  # trunk
            (3, 6),
            (6, 7),
            (7, 8),
            (8, 9),  # left arm
            (3, 10),
            (10, 11),
            (11, 12),
            (12, 13),  # right arm
            (0, 14),
            (14, 15),
            (15, 16),
            (16, 17),  # left leg
            (0, 18),
            (18, 19),
            (19, 20),
            (20, 21)
        ]  # right leg
        self.root_id = 0
        self.dataset_fps = 30
        '''
        m01 - Deep squat
        m02 - Hurdle step
        m03 - Inline lunge
        m04 - Side lunge
        m05 - Sit to stand
        m06 - Straight leg raise
        m07 - Shouler abduction
        m08 - Shoulder extension
        m09 - Shoulder rotation
        m10 - Shoulder scaption
        '''

        if is_train:
            res_ids = [0, 1, 2]
        else:
            res_ids = [3]

        self.samples = []
        self.labels = []

        idx = 0
        for filename in sorted(
                os.listdir(
                    os.path.join(self.dataset_root, 'Correct', 'Kinect',
                                 'Skeletons'))):
            if not movement in filename:
                continue

            idx = idx + 1
            if idx % 4 not in res_ids:
                continue

            filepath = os.path.join(
                os.path.join(self.dataset_root, 'Correct', 'Kinect',
                             'Skeletons', filename))
            keypoints = np.loadtxt(filepath)
            keypoints = keypoints.reshape(keypoints.shape[0], -1, 3)
            self.samples.append(keypoints)
            self.labels.append(True)

        for filename in sorted(
                os.listdir(
                    os.path.join(self.dataset_root, 'Incorrect', 'Kinect',
                                 'Skeletons'))):
            if not movement in filename:
                continue

            idx = idx + 1
            if idx % 4 not in res_ids:
                continue

            filepath = os.path.join(
                os.path.join(self.dataset_root, 'Incorrect', 'Kinect',
                             'Skeletons', filename))
            keypoints = np.loadtxt(filepath)
            keypoints = keypoints.reshape(keypoints.shape[0], -1, 3)
            self.samples.append(keypoints)
            self.labels.append(False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index] / 100.
        label = self.labels[index]

        return sample, label


class UIPRMD_Vicon_Dataset(Dataset):

    def __init__(self, dataset_root=None, movement='m01', is_train=True):
        super().__init__()
        self.dataset_root = os.path.join(dataset_root, 'data', 'UI-PRMD')
        self.is_train = is_train
        self.connectivity = [(0, 1), (1, 3), (3, 2), (2, 0), (2, 4), (3, 4),
                             (6, 9), (6, 16), (6, 7), (7, 23), (7, 24), (8, 4),
                             (4, 5), (5, 25), (5, 26), (4, 9), (9, 10),
                             (10, 11), (11, 12), (12, 14), (13, 14), (14, 15),
                             (4, 16), (16, 17), (17, 18), (18, 19), (19, 21),
                             (20, 21), (21, 22), (23, 25), (24, 26), (23, 24),
                             (25, 26), (23, 27), (27, 28), (27, 25), (28, 29),
                             (29, 30), (30, 31), (30, 32), (24, 33), (33, 34),
                             (33, 26), (34, 35), (35, 36), (36, 37), (36, 38)]
        self.root_id = (25, 26)
        self.dataset_fps = 100

        if is_train:
            res_ids = [0, 1, 2]
        else:
            res_ids = [3]

        self.samples = []
        self.labels = []

        idx = 0
        for filename in sorted(
                os.listdir(
                    os.path.join(self.dataset_root, 'Correct', 'Vicon',
                                 'Skeletons'))):
            if not movement in filename:
                continue

            idx = idx + 1
            if idx % 4 not in res_ids:
                continue
            filepath = os.path.join(
                os.path.join(self.dataset_root, 'Correct', 'Vicon',
                             'Skeletons', filename))
            keypoints = np.loadtxt(filepath)
            keypoints = keypoints.reshape(keypoints.shape[0], -1, 3)
            self.samples.append(keypoints)
            self.labels.append(True)

        for filename in sorted(
                os.listdir(
                    os.path.join(self.dataset_root, 'Incorrect', 'Vicon',
                                 'Skeletons'))):
            if not movement in filename:
                continue

            idx = idx + 1
            if idx % 4 not in res_ids:
                continue
            filepath = os.path.join(
                os.path.join(self.dataset_root, 'Incorrect', 'Vicon',
                             'Skeletons', filename))
            keypoints = np.loadtxt(filepath)
            keypoints = keypoints.reshape(keypoints.shape[0], -1, 3)
            self.samples.append(keypoints)
            self.labels.append(False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index] / 100.
        label = self.labels[index]

        return sample, label
