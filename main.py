from typing_extensions import Self
import fnet.data
import fnet.fnet_model
import json
import logging
import numpy as np
import os
import sys
import time
import torch
import wandb
from tqdm import tqdm
import datetime
import config
import gc
import pandas as pd
import tifffile
import random


def main():

    time_start = time.time()

    opts = config.get_arguments_main()

    # set random seed
    if opts.seed is not None:
        random.seed(opts.seed)
        np.random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed_all(opts.seed)

    # path init
    if not os.path.exists('exps/'):
        os.makedirs('exps/')
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

    # debugging control
    if opts.debugging or exp_name == 'integ_dataset':
        os.environ["WANDB_MODE"] = 'offline'
    else:
        os.environ["WANDB_MODE"] = 'online'

    # Setup logging
    logger = logging.getLogger('SSP')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(path_log_dir, f'run_{opts.exp_name}.log'), mode='w')
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(sh)
    logging.Formatter.converter = lambda a, b: (datetime.datetime.now() + datetime.timedelta(hours=8)).timetuple()

    # checkpoint setting
    if opts.interval_checkpoint is not None:
        checkpoint_times = int(opts.num_epochs / opts.interval_checkpoint)
        opts.epoch_checkpoint.extend([(i + 1) * opts.interval_checkpoint for i in range(checkpoint_times)])

    # wandb init
    project_name = 'SSP'
    exec('f_str = ' + opts.run_name)
    run_name = locals()['f_str']
    if opts.id is not None:
        run_id = opts.id
        os.environ["WANDB_RESUME"] = "must"
    else:
        run_id = wandb.util.generate_id()
        os.environ["WANDB_RESUME"] = "allow"
    wandb.init(
        settings=wandb.Settings(start_method="fork"),
        project=project_name,
        name=run_name,
        tags=opts.tags,
        config=opts,
        id=run_id,
        save_code=True,
        allow_val_change=True,
    )

    # save important code
    wandb.save('fnet/data/SSPdataset.py', policy='now')
    wandb.save('fnet/fnet_model.py', policy='now')
    wandb.save(f'fnet/nn_modules/{opts.nn_module}.py', policy='now')
    wandb.save('config.py', policy='now')

    # save config
    with open(os.path.join(path_log_dir, f'train_options_{opts.exp_name}.json'), 'w') as fo:
        json.dump(vars(opts), fo, indent=4, sort_keys=True)

    ###################################################################

    # load data
    logger.info('[ACTION]  Loading dataset ...')
    logger.info(f'[DATASET] Adopted dataset: {str(opts.adopted_datasets)}')
    wandb.run.summary['adopted_datasets'] = opts.adopted_datasets

    opts.adopted_datasets.sort()
    dataloader_train = fnet.get_dataloader(opts, logger, ds_type='train')
    dataloader_val = fnet.get_dataloader(opts, logger, ds_type='val')
    dataloader_test = fnet.get_dataloader(opts, logger, ds_type='test')

    logger.info('[TIME]    Elapsed time: {:.1f} s'.format(time.time() - time_start))

    ###################################################################

    # instantiate model
    logger.info('[ACTION]  Instantiating model ...')

    if opts.path_load_model is not None and os.path.exists(opts.path_load_model):
        model = fnet.load_model_from_path(opts, opts.path_load_model, gpu_ids=opts.gpu_ids)
        logger.info('[MODEL]   Model loaded from: {:s}'.format(opts.path_load_model))
    else:
        model = fnet.fnet_model.Model(
            opts=opts,
            nn_module=opts.nn_module,
            lr=opts.lr,
            gpu_ids=opts.gpu_ids,
        )
        logger.info('[MODEL]   Model initialized as: {:s}'.format(opts.nn_module))

    logger.debug(model)  # log but not print
    wandb.run.summary['model_info'] = model.__str__()  # log the info of the model
    if opts.monitor_model:  # monitor model
        wandb.watch(
            models=model.net,
            log='all',
            log_freq=100,
            log_graph=True,
        )
    logger.info('[TIME]    Elapsed time: {:.1f} s'.format(time.time() - time_start))

    ###################################################################

    # run experiment
    logger.info('[ACTION]  Start training ...')
    best_metric = np.inf
    start_epoch = model.count_epoch
    for epoch in range(start_epoch, opts.num_epochs):

        # traininig
        log_dict = run_train(opts, model, dataloader_train, epoch)

        logger.info(
            '[TRAIN]   NO.{} epcoch training | loss: {:.6f}'.format(
                epoch + 1,
                log_dict['loss/epoch'],
            ))
        wandb.log(log_dict)

        # validation
        if (epoch + 1) % opts.interval_val == 0:

            log_dict, stat_dict = run_eval(opts, model, dataloader_val, 'val', epoch)

            logger.info(
                '[VAL]     NO.{} epcoch validation | MSE: {:.6f}'.format(
                    epoch + 1,
                    log_dict['metric_val/MSE'],
                ))
            wandb.log(log_dict)

            # save checkpint
            if (epoch + 1) in opts.epoch_checkpoint:
                model_name = 'model_{}_{:04d}.p'.format(opts.exp_name, epoch + 1)
                path_save_cp_model = os.path.join(path_checkpoint_dir, model_name)
                model.save_state(path_save_cp_model)
                logger.info('[MODEL]   Checkpoint model saved to: {:s}'.format(path_save_cp_model))

            # save best model
            if log_dict['metric_val/MSE'] < best_metric:
                best_metric = log_dict['metric_val/MSE']
                model_name = 'model_best_{}.p'.format(opts.exp_name)
                path_save_best_model = os.path.join(path_checkpoint_dir, model_name)
                model.save_state(path_save_best_model)
                logger.info('[MODEL]   **Best** model saved to: {:s}'.format(path_save_best_model))
                # wandb.log(stat_dict)
                wandb.run.summary['metric_val/MSE_best@epoch'] = epoch + 1
                wandb.run.summary['metric_val/MSE_best'] = best_metric

    logger.info('[TIME]    Elapsed time: {:.1f} s'.format(time.time() - time_start))

    ###################################################################

    # release memory
    del dataloader_train
    del dataloader_val
    gc.collect()

    # re-load best model
    # dataloader_test = fnet.get_dataloader(opts, logger, ds_type='test')
    model = fnet.load_model_from_path(opts, path_save_best_model, gpu_ids=opts.gpu_ids)

    logger.info(f'[ACTION]  Evalute model: {path_save_best_model}')
    wandb.run.summary['path_eval_model'] = path_save_best_model

    # evaluate model
    log_dict, stat_dict = run_eval(opts, model, dataloader_test, 'test')

    logger.info(
        '[TEST]    Test | MSE: {:.6f}'.format(
            log_dict['metric_test/MSE'],
        ))
    wandb.log(stat_dict)
    for key in log_dict.keys():
        wandb.run.summary[key] = log_dict[key]

    if opts.save_test_preds:
        logger.info(f'[TEST]    Test predictions saved to: {opts.path_pred_dir}')
    if opts.save_test_signals_and_targets:
        logger.info(f'[TEST]    Test singals and targets saved to: {opts.path_pred_dir}')

    wandb.run.finish(quiet=True)
    logger.info('[TIME]    Elapsed time: {:.1f} s'.format(time.time() - time_start))
    logger.info('[ACTION]  Experiment ends.')


###################################################################


def run_train(opts, model, dataloader, epoch):

    time_start = time.time()
    losses = list()

    epoch_cnt_str = '[{}/{}]'.format(epoch + 1, opts.num_epochs)
    desc = '{:<10}{:<10}'.format(epoch_cnt_str, 'Training')

    for i, (signal, target, task) in enumerate(tqdm(dataloader, desc=desc)):

        model.count_iter = epoch * len(dataloader) + i + 1
        _, loss_sample = model.do_train_iter(signal, target, task)
        losses.append(loss_sample)

    model.count_epoch = epoch + 1

    comp_loss_df = pd.concat(losses)
    spec_loss_df = comp_loss_df.groupby('dataset').mean(numeric_only=True)
    final_loss_df = comp_loss_df.mean(numeric_only=True).to_frame().T

    log_dict = {'X-axis/epoch': epoch + 1}
    log_dict['loss/epoch'] = final_loss_df.iloc[0]['loss']
    for index in spec_loss_df.index:
        log_dict[f'loss_epoch/{index}'] = spec_loss_df.loc[index]['loss']
    log_dict['time/train'] = time.time() - time_start

    return log_dict


def run_eval(opts, model, dataloader, eval_type, epoch=None):

    time_start = time.time()
    metrics = list()

    if eval_type == 'val':
        epoch_cnt_str = '[{}/{}]'.format(epoch + 1, opts.num_epochs)
        phase_str = 'Validating'
    else:
        epoch_cnt_str = '[N/A]'
        phase_str = 'Testing'
    desc = '{:<10}{:<10}'.format(epoch_cnt_str, phase_str)

    for i, (signal, target, task) in enumerate(tqdm(dataloader, desc=desc)):

        info = dataloader.dataset.get_information(i)
        pred, metric_sample = model.do_eval_iter(signal, target, task, info)
        metrics.append(metric_sample)

        if eval_type == 'test' and opts.save_test_preds:
            img_id = os.path.basename(info['path_czi']).rstrip('.czi')
            path_tiff = os.path.join(opts.path_pred_dir, '{:0>3d}_pred_{}_{}.tiff'.format(i, info['dataset'], img_id))
            tifffile.imsave(path_tiff, pred.numpy()[0])
        if eval_type == 'test' and opts.save_test_signals_and_targets:
            img_id = os.path.basename(info['path_czi']).rstrip('.czi')
            path_tiff = os.path.join(opts.path_pred_dir, '{:0>3d}_signal_{}_{}.tiff'.format(i, info['dataset'], img_id))
            tifffile.imsave(path_tiff, signal.numpy()[0])
            path_tiff = os.path.join(opts.path_pred_dir, '{:0>3d}_target_{}_{}.tiff'.format(i, info['dataset'], img_id))
            tifffile.imsave(path_tiff, target.numpy()[0])

    comp_metric_df = pd.concat(metrics)
    img_id_data = ['{:0>3d}'.format(i) for i in range(len(comp_metric_df))]
    comp_metric_df.insert(loc=2, column='img_id', value=img_id_data)
    spec_metric_df = comp_metric_df.groupby('dataset').mean(numeric_only=True)
    final_metric_df = comp_metric_df.mean(numeric_only=True).to_frame().T

    log_dict = {'X-axis/epoch': epoch + 1} if eval_type == 'val' else dict()
    for column in final_metric_df:
        log_dict[f'metric_{eval_type}/{column}'] = final_metric_df.iloc[0][column]
        for index in spec_metric_df.index:
            log_dict[f'metric_{eval_type}_{column}/{index}'] = spec_metric_df.loc[index][column]

    spec_metric_df.insert(loc=0, column='dataset', value=spec_metric_df.index)
    spec_metric_df.reset_index(drop=True, inplace=True)
    stat_dict = {
        f'stat/comp_metric_{eval_type}': wandb.Table(dataframe=comp_metric_df),
        f'stat/spec_metric_{eval_type}': wandb.Table(dataframe=spec_metric_df),
        f'stat/final_metric_{eval_type}': wandb.Table(dataframe=final_metric_df),
    }

    if eval_type == 'test':
        comp_metric_df.to_csv(os.path.join(opts.path_metric_dir, f'comp_{opts.exp_name}.csv'), index=False)
        spec_metric_df.to_csv(os.path.join(opts.path_metric_dir, f'spec_{opts.exp_name}.csv'), index=False)
        final_metric_df.to_csv(os.path.join(opts.path_metric_dir, f'final_{opts.exp_name}.csv'), index=False)

    log_dict[f'time/{eval_type}'] = time.time() - time_start

    return log_dict, stat_dict


if __name__ == '__main__':
    main()
