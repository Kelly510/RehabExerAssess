import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import Graph
from .st_gcn import st_gcn


class GCNClassifier(nn.Module):

    def __init__(self,
                 num_joints=17,
                 in_channels=3,
                 num_classes=2,
                 connectivity=None,
                 strategy='Uniform',
                 max_hop=1,
                 device=None):
        super().__init__()

        self.graph = Graph(num_joints, connectivity, strategy, max_hop)
        self.A = torch.tensor(self.graph.A,
                              dtype=torch.float32,
                              requires_grad=False).to(device)

        spatial_kernel_size = self.A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * self.A.size(1))

        self.st_gcn_networks = nn.Sequential(
            st_gcn(in_channels, 64, kernel_size, 1, residual=False),
            st_gcn(64, 128, kernel_size, 1), st_gcn(128, 256, kernel_size, 1))

        self.fcn = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, keypoints_3d):
        N, T, V, C = keypoints_3d.size()
        x = keypoints_3d.permute(0, 2, 3, 1).contiguous()
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(N, C, T, V)
        x, _ = self.st_gcn_networks((x, self.A))
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1, 1, 1)

        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        return x


class RotationInvariantGCNClassifier(nn.Module):

    def __init__(self,
                 num_joints=17,
                 in_channels=17,
                 num_classes=2,
                 connectivity=None,
                 strategy='Uniform',
                 max_hop=1,
                 device=None):
        super().__init__()

        self.graph = Graph(num_joints, connectivity, strategy, max_hop)
        self.A = torch.tensor(self.graph.A,
                              dtype=torch.float32,
                              requires_grad=False).to(device)

        spatial_kernel_size = self.A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * self.A.size(1))

        self.st_gcn_networks = nn.Sequential(
            st_gcn(in_channels, 64, kernel_size, 1, residual=False),
            st_gcn(64, 128, kernel_size, 1), st_gcn(128, 256, kernel_size, 1))

        self.fcn = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, keypoints_3d):
        x = torch.einsum('ijkm,ijml->ijkl', keypoints_3d,
                         keypoints_3d.permute(0, 1, 3, 2))
        N, T, V, C = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(N, C, T, V)
        x, _ = self.st_gcn_networks((x, self.A))
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1, 1, 1)

        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        return x


class ViewAdaptiveGCNClassifier(nn.Module):

    def __init__(self,
                 num_joints=17,
                 in_channels=3,
                 num_classes=2,
                 connectivity=None,
                 strategy='Uniform',
                 max_hop=1,
                 device=None):
        super().__init__()
        self.device = device

        self.graph = Graph(num_joints, connectivity, strategy, max_hop)
        self.A = torch.tensor(self.graph.A,
                              dtype=torch.float32,
                              requires_grad=False).to(device)

        spatial_kernel_size = self.A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * self.A.size(1))

        self.st_gcn_networks = nn.Sequential(
            st_gcn(in_channels, 64, kernel_size, 1, residual=False),
            st_gcn(64, 128, kernel_size, 1), st_gcn(128, 256, kernel_size, 1))

        self.fcn = nn.Conv2d(256, num_classes, kernel_size=1)

        # View Adaptive Subnetwork
        self.view_adaptive_network = nn.ModuleList([
            st_gcn(in_channels, 16, kernel_size, 1, residual=False),
            nn.Conv2d(16, 3, kernel_size=1)
        ])

    def euler_to_rotmat_batch(self, euler_batch):
        # euler_batch: batch_size, 3
        # output: batch_size, 3, 3
        batch_size = euler_batch.shape[0]

        Rx_1 = torch.tensor([1, 0, 0]).unsqueeze(0).unsqueeze(0).repeat(
            batch_size, 1, 1).to(self.device)
        Rx_2 = torch.cat([
            torch.zeros(batch_size, 1, 1).to(self.device),
            torch.cos(euler_batch[:, 0]).reshape(batch_size, 1, 1),
            -torch.sin(euler_batch[:, 0]).reshape(batch_size, 1, 1)
        ],
                         dim=-1)
        Rx_3 = torch.cat([
            torch.zeros(batch_size, 1, 1).to(self.device),
            torch.sin(euler_batch[:, 0]).reshape(batch_size, 1, 1),
            torch.cos(euler_batch[:, 0]).reshape(batch_size, 1, 1)
        ],
                         dim=-1)
        Rx = torch.cat([Rx_1, Rx_2, Rx_3], dim=1)

        Ry_1 = torch.cat([
            torch.cos(euler_batch[:, 1]).reshape(batch_size, 1, 1),
            torch.zeros(batch_size, 1, 1).to(self.device),
            torch.sin(euler_batch[:, 1]).reshape(batch_size, 1, 1)
        ],
                         dim=-1)
        Ry_2 = torch.tensor([0, 1, 0]).unsqueeze(0).unsqueeze(0).repeat(
            batch_size, 1, 1).to(self.device)
        Ry_3 = torch.cat([
            -torch.sin(euler_batch[:, 1]).reshape(batch_size, 1, 1),
            torch.zeros(batch_size, 1, 1).to(self.device),
            torch.cos(euler_batch[:, 1]).reshape(batch_size, 1, 1)
        ],
                         dim=-1)
        Ry = torch.cat([Ry_1, Ry_2, Ry_3], dim=1)

        Rz_1 = torch.cat([
            torch.cos(euler_batch[:, 2]).reshape(batch_size, 1, 1),
            -torch.sin(euler_batch[:, 2]).reshape(batch_size, 1, 1),
            torch.zeros(batch_size, 1, 1).to(self.device)
        ],
                         dim=-1)
        Rz_2 = torch.cat([
            torch.sin(euler_batch[:, 2]).reshape(batch_size, 1, 1),
            torch.cos(euler_batch[:, 2]).reshape(batch_size, 1, 1),
            torch.zeros(batch_size, 1, 1).to(self.device)
        ],
                         dim=-1)
        Rz_3 = torch.tensor([0, 0, 1]).unsqueeze(0).unsqueeze(0).repeat(
            batch_size, 1, 1).to(self.device)
        Rz = torch.cat([Rz_1, Rz_2, Rz_3], dim=1)

        rotmat = torch.einsum('nvc,ncv,nvc->nvc', Rx, Ry, Rz)
        return rotmat

    def forward(self, keypoints_3d):
        N, T, V, C = keypoints_3d.size()

        va_feature, _ = self.view_adaptive_network[0](
            (keypoints_3d.permute(0, 3, 1, 2).contiguous(), self.A))
        va_feature = F.avg_pool2d(va_feature, va_feature.size()[2:])
        va_feature = va_feature.view(N, -1, 1, 1)
        output = self.view_adaptive_network[1](va_feature)
        output = output.view(output.size(0), -1)
        rotmat = self.euler_to_rotmat_batch(output)
        keypoints_3d = torch.einsum('ntvc,ncj->ntvj', keypoints_3d, rotmat)

        x = keypoints_3d.permute(0, 2, 3, 1).contiguous()
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(N, C, T, V)
        x, _ = self.st_gcn_networks((x, self.A))
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1, 1, 1)

        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        return x
