import os
import yaml
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from mvn.dataset import SingleMoveDataset
from mvn.classifier import GCNClassifier, RotationInvariantGCNClassifier, ViewAdaptiveGCNClassifier

from utils.arguments import get_args
from utils.setup import setup_seed, setup_experiment
from utils.plot_curve import plot_train_curve
from mvn.cam import visualize_CAM


def one_move(movement, exp_dir, result_dict, device, logger, args):
    movement_dir = os.path.join(exp_dir, movement)
    os.makedirs(movement_dir, exist_ok=True)
    vis_dir = os.path.join(movement_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    if movement.startswith('IRDS'):
        num_joints = 25
    elif movement.startswith('UIPRMD_Kinect'):
        num_joints = 22
    elif movement.startswith('UIPRMD_Vicon'):
        num_joints = 39
    elif movement.startswith('PushUp'):
        num_joints = 17

    loss_function = nn.CrossEntropyLoss()
    train_dataset = SingleMoveDataset(dataset_root=args.root,
                                      movement_class=movement,
                                      norm_orient=args.Pnorm,
                                      is_train=True)
    train_dataloader = DataLoader(train_dataset,
                                  args.batch_size,
                                  shuffle=True,
                                  num_workers=1)
    valid_dataset = SingleMoveDataset(dataset_root=args.root,
                                      movement_class=movement,
                                      norm_orient=args.Pnorm,
                                      is_train=False)
    valid_dataloader = DataLoader(valid_dataset,
                                  args.batch_size,
                                  shuffle=False,
                                  num_workers=1)
    train_num, valid_num = len(train_dataset), len(valid_dataset)
    logger.info('Dataset: train {}, valid {}'.format(train_num, valid_num))

    model = {
        'gcn':
        GCNClassifier(num_joints=num_joints,
                      in_channels=3,
                      connectivity=train_dataset.connectivity,
                      strategy=args.strategy,
                      device=device),
        'ri-gcn':
        RotationInvariantGCNClassifier(num_joints=num_joints,
                                       in_channels=num_joints,
                                       connectivity=train_dataset.connectivity,
                                       strategy=args.strategy,
                                       device=device),
        'va-gcn':
        ViewAdaptiveGCNClassifier(num_joints=num_joints,
                                  in_channels=3,
                                  connectivity=train_dataset.connectivity,
                                  strategy=args.strategy,
                                  device=device)
    }[args.model].to(device)
    optimizer = Adam(model.parameters(), args.lr)

    losses_epoch = []
    accuracy_epoch = []
    recall_epoch = []
    precision_epoch = []
    best_accuracy = 0
    for i in range(args.epoch):
        losses = []
        model.train()
        for j, (samples_batch, labels_batch) in enumerate(train_dataloader):
            optimizer.zero_grad()

            samples_batch = samples_batch.float().to(device)
            labels_batch = labels_batch.to(device)
            labels_batch_binary = (labels_batch !=
                                   0).long()  # incorrect - 1, correct - 0

            label_pred = model(samples_batch)
            loss = loss_function(label_pred, labels_batch_binary)

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        mean_loss = np.mean(losses)
        losses_epoch.append(mean_loss)

        accuracy, recall, precision = evaluate(model, valid_dataloader, device)
        accuracy_epoch.append(accuracy)
        recall_epoch.append(recall)
        precision_epoch.append(precision)

        if accuracy > best_accuracy:
            torch.save(model.state_dict(),
                       os.path.join(movement_dir, 'model.pth'))
            best_accuracy = accuracy

    plot_train_curve(losses_epoch, accuracy_epoch, recall_epoch,
                     precision_epoch, movement_dir)

    checkpoint = torch.load(os.path.join(movement_dir, 'model.pth'),
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)
    accuracy, recall, precision = evaluate(model, valid_dataloader, device, vis_matrix=True, \
        save_path=os.path.join(movement_dir, 'confusion_matrix.jpg'))
    OvR = visualize_CAM(model, valid_dataloader, device, vis_dir, args.vis,
                        movement_dir)

    logger.info(
        'Accuracy: {:.4f}, recall: {:.4f}, precision: {:.4f}, OvR: {:.4f}'.
        format(accuracy, recall, precision, OvR))

    result_dict[movement]['accuracy'] = float(accuracy)
    result_dict[movement]['recall'] = float(recall)
    result_dict[movement]['precision'] = float(precision)
    result_dict[movement]['OvR'] = float(OvR)

    return result_dict


def train(args):
    device = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
    exp_dir, logger = setup_experiment(args)
    logger.info(args)

    result_dict = defaultdict(dict)

    if args.dataset == 'PushUp':
        one_move('PushUp_m01', exp_dir, result_dict, device, logger, args)

    elif args.dataset == 'IRDS':
        for movement in [
                'IRDS_m01', 'IRDS_m02', 'IRDS_m03', 'IRDS_m04', 'IRDS_m05',
                'IRDS_m06', 'IRDS_m07', 'IRDS_m08', 'IRDS_m09'
        ]:
            result_dict = one_move(movement, exp_dir, result_dict, device,
                                   logger, args)

    elif args.dataset == 'UIPRMD_Kinect':
        for movement in [
                'UIPRMD_Kinect_m01', 'UIPRMD_Kinect_m02', 'UIPRMD_Kinect_m03',
                'UIPRMD_Kinect_m04', 'UIPRMD_Kinect_m05', 'UIPRMD_Kinect_m06',
                'UIPRMD_Kinect_m07', 'UIPRMD_Kinect_m08', 'UIPRMD_Kinect_m09',
                'UIPRMD_Kinect_m10'
        ]:
            result_dict = one_move(movement, exp_dir, result_dict, device,
                                   logger, args)

    elif args.dataset == 'UIPRMD_Vicon':
        for movement in [
                'UIPRMD_Vicon_m01', 'UIPRMD_Vicon_m02', 'UIPRMD_Vicon_m03',
                'UIPRMD_Vicon_m04', 'UIPRMD_Vicon_m05', 'UIPRMD_Vicon_m06',
                'UIPRMD_Vicon_m07', 'UIPRMD_Vicon_m08', 'UIPRMD_Vicon_m09',
                'UIPRMD_Vicon_m10'
        ]:
            result_dict = one_move(movement, exp_dir, result_dict, device,
                                   logger, args)

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

    with open(os.path.join(exp_dir, 'result.yaml'), 'w+',
              encoding='utf-8') as f:
        yaml.dump(dict(result_dict), f, allow_unicode=True, sort_keys=False)


def evaluate(model, dataloader, device, vis_matrix=False, save_path=None):
    model.eval()
    labels_gt = []
    labels_pred = []
    for j, (samples_batch, labels_batch) in enumerate(dataloader):
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
    recall = metrics.recall_score(y_true=labels_gt,
                                  y_pred=labels_pred)  #, zero_division=0)
    precision = metrics.precision_score(y_true=labels_gt,
                                        y_pred=labels_pred,
                                        zero_division=0)

    if vis_matrix:
        matrix = metrics.confusion_matrix(y_true=labels_gt, y_pred=labels_pred)
        plt.matshow(matrix, cmap=plt.cm.Reds)
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                plt.annotate(matrix[j, i],
                             xy=(i, j),
                             horizontalalignment='center',
                             verticalalignment='center')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(save_path)

        plt.clf()
        plt.close('all')

    return accuracy, recall, precision


if __name__ == '__main__':
    args = get_args()
    setup_seed(args.seed)
    train(args)
