import os
import csv
import numpy as np
from torch.utils.data import Dataset


class PushUp_Dataset(Dataset):

    def __init__(self, dataset_root=None, movement=None, is_train=True):

        super().__init__()
        self.dataset_root = os.path.join(dataset_root, 'data', 'PushUp')
        self.is_train = is_train
        self.connectivity = [(0, 1), (1, 2), (2, 6), (5, 4), (4, 3), (3, 6),
                             (6, 7), (7, 8), (8, 16), (9, 16), (8, 12),
                             (11, 12), (10, 11), (8, 13), (13, 14), (14, 15)]
        self.root_id = 6
        self.dataset_fps = 32

        sequence_info = []
        with open(os.path.join(self.dataset_root, 'info.CSV')) as f:
            csv_data = csv.reader(f)
            for i, line in enumerate(csv_data):
                if i == 0:
                    continue
                sequence_info.append(line)

        if is_train:
            subject_group = [
                'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10',
                'S11', 'S12'
            ]
        else:
            subject_group = ['S13', 'S14', 'S15', 'S16']

        self.samples = []
        for i, item in enumerate(sequence_info):
            if item[0] not in subject_group:
                continue

            kpts_path = os.path.join(self.dataset_root, item[0], item[1],
                                     'keypoints_3d.npy')
            keypoints_3d = np.load(kpts_path)

            label_path = os.path.join(self.dataset_root, item[0], item[1],
                                      'labels.txt')
            with open(label_path, 'r') as f:
                labels = []
                for line in f.readlines():
                    labels.append(list(map(int, line.split())))

            for j in range(len(labels)):
                fragment = keypoints_3d[labels[j][0]:labels[j][1] + 1]
                item_info = [labels[j], fragment]
                self.samples.append(item_info)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        label, keypoints = self.samples[index]
        label = label[2]
        rotmat = np.array([[1.0000000, 0.0000000, 0.0000000],
                           [0.0000000, 0.0000000, -1.0000000],
                           [0.0000000, 1.0000000, 0.0000000]])
        keypoints = np.matmul(keypoints, rotmat) / 2.
        return keypoints, label
