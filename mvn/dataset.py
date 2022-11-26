import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

from .datasets.IRDS import IRDS_Dataset
from .datasets.UIPRMD import UIPRMD_Kinect_Dataset, UIPRMD_Vicon_Dataset
from .datasets.PushUp import PushUp_Dataset


class SingleMoveDataset(Dataset):

    def __init__(self,
                 dataset_root=None,
                 movement_class='IRDS_m01',
                 norm_length=True,
                 norm_orient=False,
                 aug_angle=0,
                 is_train=True):

        super().__init__()

        self.dataset_name = '_'.join(movement_class.split('_')[:-1])
        self.movement = movement_class.split('_')[-1]
        self.norm_length = norm_length
        self.norm_orient = norm_orient
        self.aug_angle = aug_angle
        self.is_train = is_train

        self.dataset = {
            'IRDS': IRDS_Dataset,
            'UIPRMD_Kinect': UIPRMD_Kinect_Dataset,
            'UIPRMD_Vicon': UIPRMD_Vicon_Dataset,
            'PushUp': PushUp_Dataset
        }[self.dataset_name](dataset_root=dataset_root,
                             movement=self.movement,
                             is_train=is_train)

        self.dataset_fps = self.dataset.dataset_fps
        self.root_id = self.dataset.root_id
        self.connectivity = self.dataset.connectivity

    def __len__(self):
        return len(self.dataset)

    def _sampling(self, sample):
        if sample.shape[0] < self.dataset_fps:
            end_frame = sample[-2:-1].repeat(self.dataset_fps -
                                             sample.shape[0],
                                             axis=0)
            keypoints_sampled = np.concatenate([sample, end_frame], axis=0)
        else:
            ratio = sample.shape[0] / self.dataset_fps
            indexes = (np.array(range(self.dataset_fps)) * ratio).astype(
                np.int32)
            keypoints_sampled = sample[indexes]
        return keypoints_sampled

    def _get_major(self, coord):
        hist, bins = np.histogram(coord)
        major_idx = np.argmax(hist)
        major_ran = (bins[major_idx], bins[major_idx + 1])
        major_list = [
            item for item in coord
            if item >= major_ran[0] and item <= major_ran[1]
        ]
        major = np.mean(major_list)
        return major

    def _centralize(self, sample):
        center = np.repeat(np.mean(sample, axis=1, keepdims=True),
                           sample.shape[1],
                           axis=1)
        sample = sample - center
        return sample

    def _normalize_orient(self, sample):

        def fit_plane_rotmat(points, target):
            # reference: https://blog.csdn.net/Dontla/article/details/108445277
            z = points[:, 2]
            A = np.stack([points[:, 0], points[:, 1], np.ones_like(z)], axis=1)
            params = np.matmul(np.linalg.pinv(A), z)
            norm_vector = np.array([params[0], params[1], -1])
            norm_vector = norm_vector / np.linalg.norm(norm_vector)
            rot_vector = np.cross(norm_vector, target)
            inner = np.dot(norm_vector, target)
            length = np.arcsin(
                np.linalg.norm(rot_vector)
            ) if inner >= 0 else np.pi - np.arcsin(np.linalg.norm(rot_vector))
            rot_vector = rot_vector / np.linalg.norm(rot_vector) * length
            rot_mat = R.from_rotvec(rot_vector, degrees=False).as_matrix()
            return rot_mat

        pelvis = np.mean(sample[:, (6, 2, 3), :], axis=0)
        rotation = fit_plane_rotmat(pelvis, np.array([1, 0, 0]))
        sample = np.einsum('ij,kmj->kmi', rotation, sample)
        return sample

    def _augment(self, sample):
        euler = (2 * np.random.rand(3) - 1) * (self.aug_angle / 180.) * np.pi
        rotation = R.from_euler('zxy', euler, degrees=False).as_matrix()
        sample = np.einsum('ij,kmj->kmi', rotation, sample)
        return sample

    def __getitem__(self, index):
        '''
        label = 1, correct
        label = 0, incorrect
        '''
        sample, label = self.dataset.__getitem__(index)

        if self.norm_length:
            sample = self._sampling(sample)

        sample = self._centralize(sample)

        if self.norm_orient:
            sample = self._normalize_orient(sample)

        if self.aug_angle != 0:
            sample = self._augment(sample)

        return sample, label
