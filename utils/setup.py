import os
import random
import logging
import numpy as np

import torch


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_experiment(args):
    dataset_prefix = args.dataset
    model_prefix = {
        'gcn': 'GCN',
        'ri-gcn': 'RIGCN',
        'va-gcn': 'VAGCN'
    }[args.model]
    preproc_prefix = 'Pnorm' if args.Pnorm else 'none'
    seed_prefix = 'seed{}'.format(args.seed)
    exp_dir = os.path.join(
        './logs',
        '_'.join([dataset_prefix, model_prefix, preproc_prefix, seed_prefix]))

    os.makedirs(exp_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(os.path.join(exp_dir, 'log.txt'))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)

    return exp_dir, logger
