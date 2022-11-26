import os
import numpy as np
from sklearn import metrics
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from mvn.dataset import SingleMoveDataset
from mvn.classifier import GCNClassifier, RotationInvariantGCNClassifier, ViewAdaptiveGCNClassifier

from utils.arguments import get_args
from utils.setup import setup_experiment, setup_seed
from mvn.cam import visualize_CAM


def one_move(movement, exp_dir, result_dict, device, logger, args, aug_angle):
    movement_dir = os.path.join(exp_dir, movement)
    vis_dir = os.path.join(movement_dir, 'vis')

    if movement.startswith('IRDS'):
        num_joints = 25
    elif movement.startswith('UIPRMD_Kinect'):
        num_joints = 22
    elif movement.startswith('UIPRMD_Vicon'):
        num_joints = 39
    elif movement.startswith('PushUp'):
        num_joints = 17

    valid_dataset = SingleMoveDataset(dataset_root=args.root,
                                      movement_class=movement,
                                      norm_orient=args.Pnorm,
                                      aug_angle=aug_angle,
                                      is_train=False)
    valid_dataloader = DataLoader(valid_dataset,
                                  args.batch_size,
                                  shuffle=False,
                                  num_workers=1)

    model = {
        'gcn':
        GCNClassifier(num_joints=num_joints,
                      in_channels=3,
                      connectivity=valid_dataset.connectivity,
                      strategy=args.strategy,
                      device=device),
        'ri-gcn':
        RotationInvariantGCNClassifier(num_joints=num_joints,
                                       in_channels=num_joints,
                                       connectivity=valid_dataset.connectivity,
                                       strategy=args.strategy,
                                       device=device),
        'va-gcn':
        ViewAdaptiveGCNClassifier(num_joints=num_joints,
                                  in_channels=3,
                                  connectivity=valid_dataset.connectivity,
                                  strategy=args.strategy,
                                  device=device)
    }[args.model].to(device)

    checkpoint = torch.load(os.path.join(movement_dir, 'model.pth'),
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)

    model.eval()
    labels_gt = []
    labels_pred = []
    for j, (samples_batch, labels_batch) in enumerate(valid_dataloader):
        samples_batch = samples_batch.float().to(device)
        labels_batch_binary = labels_batch != 0  # incorrect - 1, correct - 0

        with torch.no_grad():
            label_pred = model(samples_batch)

        label_pred = torch.argmax(label_pred, dim=1)
        labels_gt.append(np.array(labels_batch_binary))
        labels_pred.append(np.array(label_pred.cpu()))

    labels_gt = np.concatenate(labels_gt, axis=0)
    labels_pred = np.concatenate(labels_pred, axis=0)
    accuracy = metrics.accuracy_score(y_true=labels_gt, y_pred=labels_pred)
    recall = metrics.recall_score(y_true=labels_gt, y_pred=labels_pred)
    precision = metrics.precision_score(y_true=labels_gt,
                                        y_pred=labels_pred,
                                        zero_division=0)

    OvR = visualize_CAM(model, valid_dataloader, device, vis_dir, args.vis,
                        movement_dir)

    logger.info(
        '{}: accuracy {:.4f}, recall {:.4f}, precision {:.4f}, OvR {:.4f}'.
        format(movement, accuracy, recall, precision, OvR))

    result_dict[movement]['accuracy'] = float(accuracy)
    result_dict[movement]['recall'] = float(recall)
    result_dict[movement]['precision'] = float(precision)
    result_dict[movement]['OvR'] = float(OvR)

    return result_dict


def inference(args):
    device = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
    exp_dir, logger = setup_experiment(args)
    logger.info(args)

    for aug_angle in args.aug_angle if isinstance(args.aug_angle,
                                                  list) else [args.aug_angle]:

        result_dict = defaultdict(dict)
        movement_list = {
            'PushUp': ['PushUp_m01'],
            'IRDS': [
                'IRDS_m01', 'IRDS_m02', 'IRDS_m03', 'IRDS_m04', 'IRDS_m05',
                'IRDS_m06', 'IRDS_m07', 'IRDS_m08', 'IRDS_m09'
            ],
            'UIPRMD_Kinect': [
                'UIPRMD_Kinect_m01', 'UIPRMD_Kinect_m02', 'UIPRMD_Kinect_m03',
                'UIPRMD_Kinect_m04', 'UIPRMD_Kinect_m05', 'UIPRMD_Kinect_m06',
                'UIPRMD_Kinect_m07', 'UIPRMD_Kinect_m08', 'UIPRMD_Kinect_m09',
                'UIPRMD_Kinect_m10'
            ],
            'UIPRMD_Vicon': [
                'UIPRMD_Vicon_m01', 'UIPRMD_Vicon_m02', 'UIPRMD_Vicon_m03',
                'UIPRMD_Vicon_m04', 'UIPRMD_Vicon_m05', 'UIPRMD_Vicon_m06',
                'UIPRMD_Vicon_m07', 'UIPRMD_Vicon_m08', 'UIPRMD_Vicon_m09',
                'UIPRMD_Vicon_m10'
            ]
        }[args.dataset]

        for movement in movement_list:
            result_dict = one_move(movement, exp_dir, result_dict, device,
                                   logger, args, aug_angle)

        accuracy_list = []
        recall_list = []
        precision_list = []
        OvR_list = []
        for move in result_dict.keys():
            accuracy_list.append(result_dict[move]['accuracy'])
            recall_list.append(result_dict[move]['recall'])
            precision_list.append(result_dict[move]['precision'])
            OvR_list.append(result_dict[move]['OvR'])

        result_dict['mean']['accuracy'] = float(np.mean(accuracy_list))
        result_dict['mean']['recall'] = float(np.mean(recall_list))
        result_dict['mean']['precision'] = float(np.mean(precision_list))
        result_dict['mean']['OvR'] = float(np.mean(OvR_list))

        logger.info('disturb angle {}: mean accuracy {}'.format(
            aug_angle, result_dict['mean']['accuracy']))


if __name__ == '__main__':
    args = get_args()
    setup_seed(args.seed)
    inference(args)
