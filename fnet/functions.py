import importlib
import json
import os
import pdb
import sys
import fnet
import torch


def load_model(path_model, gpu_ids=0, module='fnet_model'):
    module_fnet_model = importlib.import_module('fnet.' + module)
    if os.path.isdir(path_model):
        path_model = os.path.join(path_model, 'model.p')
    model = module_fnet_model.Model()
    model.load_state(path_model, gpu_ids=gpu_ids)
    return model


def load_model_cus(path_model, gpu_ids=0, module='fnet_model'):
    module_fnet_model = importlib.import_module('fnet.' + module)
    if os.path.isdir(path_model):
        name_model = os.path.basename(path_model)
        path_model = os.path.join(path_model, f'{name_model}_model.p')
    model = module_fnet_model.Model()
    model.load_state(path_model, gpu_ids=gpu_ids)
    return model


def load_model_from_dir(path_model_dir, gpu_ids=0):
    if path_model_dir[-1] != '_':  # NOTE cus setting
        path_model_state = os.path.join(path_model_dir, 'model.p')
    else:
        path_model_state = path_model_dir + 'model.p'
    model = fnet.fnet_model.Model()
    model.load_state(path_model_state, gpu_ids=gpu_ids)
    return model


def load_model_from_path(opts, path_model_state, gpu_ids=0):
    model = fnet.fnet_model.Model(opts, gpu_ids=gpu_ids)
    model.load_state(path_model_state, gpu_ids=gpu_ids)
    return model


def get_dataloader(opts, logger, ds_type):
    dataset = getattr(fnet.data, opts.class_dataset)(opts, logger, ds_type)
    batch_size = opts.batch_size if ds_type == 'train' else 1
    if_shuffle = True if ds_type == 'train' else False
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=if_shuffle,
        num_workers=opts.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    return dataloader



