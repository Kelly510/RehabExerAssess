import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument('--root',
                        type=str,
                        default='./',
                        help='path to root of this project')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--vis',
                        action='store_true',
                        help='visualize movement CAM')

    # dataset and preprocess
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['IRDS', 'UIPRMD_Kinect', 'UIPRMD_Vicon', 'PushUp'],
        help='dataset type')
    parser.add_argument('--Pnorm',
                        action='store_true',
                        help='normalize orientation of skeletons')
    parser.add_argument('--aug_angle',
                        type=int,
                        nargs='+',
                        default=0,
                        help='rotation augmentation for PushUp dataset')

    # classification model
    parser.add_argument('--model',
                        type=str,
                        default='ri-gcn',
                        choices=['gcn', 'ri-gcn', 'va-gcn'],
                        help='type of model to be used')
    parser.add_argument('--strategy',
                        type=str,
                        choices=['uniform', 'distance'],
                        default='uniform',
                        help='strategy of constructing A matrix in ST-GCN')

    # training hyper parameters
    parser.add_argument('--epoch',
                        type=int,
                        default=100,
                        help='epochs to train')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='batch size of data')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    args = parser.parse_args()
    return args
