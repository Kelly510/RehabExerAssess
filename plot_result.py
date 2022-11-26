import os
import yaml
import numpy as np
from matplotlib import pyplot as plt


def load_result(path, dataset):
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    accuracy_dict = {}
    recall_dict = {}
    precision_dict = {}
    overlap_ratio_dict = {}

    for movement, metrics in data.items():
        accuracy_dict[movement] = metrics['accuracy']
        recall_dict[movement] = metrics['recall']
        precision_dict[movement] = metrics['precision']
        overlap_ratio_dict[movement] = metrics['OvR']

    return accuracy_dict, recall_dict, precision_dict, overlap_ratio_dict


def plot_bar(metric_dict_1,
             metric_dict_2,
             metric_dict_3,
             metric_dict_4,
             save_fig,
             ylim=[0.0, 1.1]):
    plt.figure(figsize=(20, 6), dpi=100)
    total_width, n = 0.8, 4
    width = total_width / n

    ticks = list(metric_dict_1.keys())
    x = np.arange(len(ticks))
    p1 = plt.bar(x - width * 1.5,
                 list(metric_dict_1.values()),
                 width=width,
                 tick_label=ticks,
                 label='vinalla')
    plt.bar_label(p1, fmt='%.4f', label_type='edge', fontsize=6)

    p2 = plt.bar(x - width * 0.5,
                 list(metric_dict_2.values()),
                 width=width,
                 tick_label=ticks,
                 label='VAGCN')
    plt.bar_label(p2, fmt='%.4f', label_type='edge', fontsize=6)

    p3 = plt.bar(x + width * 0.5,
                 list(metric_dict_3.values()),
                 width=width,
                 tick_label=ticks,
                 label='Pnorm')
    plt.bar_label(p3, fmt='%.4f', label_type='edge', fontsize=6)

    p4 = plt.bar(x + width * 1.5,
                 list(metric_dict_4.values()),
                 width=width,
                 tick_label=ticks,
                 label='w.t. RI')
    plt.bar_label(p4, fmt='%.4f', label_type='edge', fontsize=6)

    plt.xticks(x, labels=ticks)
    plt.ylim(ylim)
    plt.legend(loc=3)
    plt.savefig(os.path.join('figures', save_fig))


if __name__ == '__main__':
    dataset = 'PushUp'
    seed = 1
    accuracy_dict_1, recall_dict_1, precision_dict_1, overlap_ratio_dict_1 = load_result(
        'logs/{}_GCN_none_seed{}/result.yaml'.format(dataset, seed), dataset)
    accuracy_dict_2, recall_dict_2, precision_dict_2, overlap_ratio_dict_2 = load_result(
        'logs/{}_VAGCN_none_seed{}/result.yaml'.format(dataset, seed), dataset)
    accuracy_dict_3, recall_dict_3, precision_dict_3, overlap_ratio_dict_3 = load_result(
        'logs/{}_GCN_Pnorm_seed{}/result.yaml'.format(dataset, seed), dataset)
    accuracy_dict_4, recall_dict_4, precision_dict_4, overlap_ratio_dict_4 = load_result(
        'logs/{}_RIGCN_none_seed{}/result.yaml'.format(dataset, seed), dataset)

    plot_bar(accuracy_dict_1, accuracy_dict_2, accuracy_dict_3,
             accuracy_dict_4, 'accuracy_{}.png'.format(dataset))
    plot_bar(recall_dict_1, recall_dict_2, recall_dict_3, recall_dict_4,
             'recall_{}.png'.format(dataset))
    plot_bar(precision_dict_1, precision_dict_2, precision_dict_3,
             precision_dict_4, 'precision_{}.png'.format(dataset))
    plot_bar(overlap_ratio_dict_1, overlap_ratio_dict_2, overlap_ratio_dict_3,
             overlap_ratio_dict_4, 'OvR_{}.png'.format(dataset))
