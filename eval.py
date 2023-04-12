import argparse
from statistics import mean

from regex import E
import fnet.data
import fnet.fnet_model
import json
import logging
import numpy as np
import os
import pdb
import sys
import time
import torch
import warnings
import wandb
from tqdm import tqdm
from torchvision import transforms
import datetime
import config
import gc
import pandas as pd
import tifffile
from main import run_eval


def main():

    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument(
        '--adopted_datasets',
        nargs='+',
        default=[
            'alpha_tubulin',
            'beta_actin',
            'desmoplakin',
            'dna',
            'fibrillarin',
            'lamin_b1',
            'membrane_caax_63x',
            'myosin_iib',
            'sec61_beta',
            'st6gal1',
            'tom20',
            'zo1',
        ],
        help='list of the names of adopted datasets'
    )
    parser.add_argument('--class_dataset', default='SSPDataset', help='Dataset class')

    # training
    parser.add_argument('--nn_module', default='RepMode', help='name of neural network module')
    parser.add_argument('--batch_size_eval', type=int, default=8, help='size of each batch for evaluation')

    # path
    parser.add_argument('--path_exp_dir', type=str, help='directory for saving exp stuff')
    parser.add_argument('--path_dataset_csv', type=str, default='data/csvs', help='path to csv for constructing dataset')
    parser.add_argument('--path_dataset_czi', type=str, default='data', help='path to czi images of datasets')
    parser.add_argument('--path_load_dataset', type=str, help='path to load the dataset')
    parser.add_argument('--path_save_dataset', type=str, help='path to save the dataset')
    parser.add_argument('--path_load_model', type=str, help='path to load the model')

    # device & seed
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=0, help='GPU ID')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--num_workers', default=10, type=int, help='number of workers for data loading')

    # state
    parser.add_argument('--save_test_preds', action='store_true', help='set to save predicted results in test')
    parser.add_argument('--save_test_signals_and_targets', action='store_true', help='set to save signals and targets in test')

    # log
    parser.add_argument('--id', type=str, help='id for logging')
    opts = parser.parse_args()

    time_start = time.time()

    # set random seed
    if opts.seed is not None:
        np.random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed_all(opts.seed)

    # path init
    exp_name = os.path.basename(opts.path_exp_dir)
    setattr(opts, 'exp_name', exp_name)
    if not os.path.exists(opts.path_exp_dir):
        os.makedirs(opts.path_exp_dir)
    path_log_dir = os.path.join(opts.path_exp_dir, 'logs')
    if not os.path.exists(path_log_dir):
        os.makedirs(path_log_dir)
    path_checkpoint_dir = os.path.join(opts.path_exp_dir, 'checkpoints')
    if not os.path.exists(path_checkpoint_dir):
        os.makedirs(path_checkpoint_dir)
    path_metric_dir = os.path.join(opts.path_exp_dir, 'metrics')
    if not os.path.exists(path_metric_dir):
        os.makedirs(path_metric_dir)
    setattr(opts, 'path_metric_dir', path_metric_dir)
    path_pred_dir = os.path.join(opts.path_exp_dir, 'preds')
    if not os.path.exists(path_pred_dir):
        os.makedirs(path_pred_dir)
    setattr(opts, 'path_pred_dir', path_pred_dir)

    # Setup logging
    logger = logging.getLogger('FluorPred')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(path_log_dir, f'run_{opts.exp_name}.log'), mode='a')  # use 'a' mode
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(sh)
    logging.Formatter.converter = lambda a, b: (datetime.datetime.now() + datetime.timedelta(hours=8)).timetuple()

    # wandb init
    if opts.id is not None:
        run_id = opts.id
        os.environ["WANDB_RESUME"] = "must"
        wandb.init(
            settings=wandb.Settings(start_method="fork"),
            id=run_id,
        )
        wandb.run.summary['path_eval_model'] = opts.path_load_model

    ###################################################################

    # load data
    logger.info('[ACTION]  Loading dataset ...')
    opts.adopted_datasets.sort()
    logger.info(f'[DATASET] Adopted dataset: {str(opts.adopted_datasets)}')
    dataloader_test = fnet.get_dataloader(opts, logger, ds_type='test')

    logger.info('[TIME]    Elapsed time: {:.1f} s'.format(time.time() - time_start))

    ###################################################################

    # instantiate model
    logger.info('[ACTION]  Instantiating model ...')
    model = fnet.load_model_from_path(opts, opts.path_load_model, gpu_ids=opts.gpu_ids)
    logger.info('[MODEL]   Model loaded from: {:s}'.format(opts.path_load_model))

    logger.info('[TIME]    Elapsed time: {:.1f} s'.format(time.time() - time_start))

    ###################################################################

    # run eval
    logger.info(f'[ACTION]  Evalute model: {opts.path_load_model}')

    log_dict, stat_dict = run_eval(opts, model, dataloader_test, 'test')

    logger.info(
        '[TEST]    Test | MSE: {:.6f}'.format(
            log_dict['metric_test/MSE'],
        ))
    if opts.id is not None:
        wandb.log(stat_dict)
        for key in log_dict.keys():
            wandb.run.summary[key] = log_dict[key]

    if opts.save_test_preds:
        logger.info(f'[TEST]    Test predictions saved to: {opts.path_pred_dir}')
    if opts.save_test_signals_and_targets:
        logger.info(f'[TEST]    Test singals and targets saved to: {opts.path_pred_dir}')

    logger.info('[TIME]    Elapsed time: {:.1f} s'.format(time.time() - time_start))
    logger.info('[ACTION]  Evalution ends.')


if __name__ == '__main__':
    main()
