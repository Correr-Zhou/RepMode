from builtins import list
import enum
import os
import numpy as np
import torch
import importlib
import pdb
from torch.cuda.amp import GradScaler, autocast
import wandb
from fnet.metric import *
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
from math import ceil


class Model(object):
    def __init__(
            self,
            opts,
            nn_module=None,
            init_weights=True,
            lr=0.001,
            criterion_fn=torch.nn.MSELoss,
            gpu_ids=-1,
    ):
        self.opts = opts
        self.nn_module = nn_module
        self.init_weights = init_weights
        self.lr = lr
        self.count_iter = 0
        self.count_epoch = 0
        self.gpu_ids = [gpu_ids] if isinstance(gpu_ids, int) else gpu_ids
        self.device = torch.device('cuda', self.gpu_ids[0]) if self.gpu_ids[0] >= 0 else torch.device('cpu')
        self.patch_size = (32, 128, 128)

        self.criterion = criterion_fn(reduction='none')

        self._init_model()

        if len(self.gpu_ids) > 1:
            self.net = torch.nn.DataParallel(
                self.net,
                device_ids=self.gpu_ids,
            )

        self.scaler = GradScaler()

    def _init_model(self):
        if self.nn_module is None:
            self.net = None
            return
        self.net = importlib.import_module('fnet.nn_modules.' + self.nn_module).Net(self.opts)
        self.net.to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def get_state(self):
        return dict(
            nn_module=self.nn_module,
            opts=self.opts,
            nn_state=self.net.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            count_iter=self.count_iter,
            count_epoch=self.count_epoch,
        )

    def to_gpu(self, gpu_ids):
        if isinstance(gpu_ids, int):
            gpu_ids = [gpu_ids]
        self.gpu_ids = gpu_ids
        self.device = torch.device('cuda', self.gpu_ids[0]) if self.gpu_ids[0] >= 0 else torch.device('cpu')
        self.net.to(self.device)
        _set_gpu_recursive(self.optimizer.state, self.gpu_ids[0])

    def save_state(self, path_save):
        curr_gpu_ids = self.gpu_ids
        dirname = os.path.dirname(path_save)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self.to_gpu(-1)
        torch.save(self.get_state(), path_save)
        self.to_gpu(curr_gpu_ids)

    def load_state(self, path_load, gpu_ids=-1):
        state_dict = torch.load(path_load)
        self.nn_module = state_dict['nn_module']
        self.opts = state_dict['opts']
        self.opts.gpu_ids = gpu_ids
        self._init_model()
        self.net.load_state_dict(state_dict['nn_state'])
        self.optimizer.load_state_dict(state_dict['optimizer_state'])
        self.count_iter = state_dict['count_iter']
        self.count_epoch = state_dict['count_epoch']
        self.to_gpu(gpu_ids)

    def do_train_iter(self, signal, target, task):

        signal = signal.to(self.device)
        target = target.to(self.device)
        task = task.to(self.device)

        self.net.train()

        # AMP training
        self.optimizer.zero_grad()
        with autocast():
            output = self.net(signal, task)
            loss_nomean = self.criterion(output, target)
            loss = torch.mean(loss_nomean)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        log_dict = {
            'X-axis/iter': self.count_iter,
            'loss/iter': loss.item(),
        }
        loss_diff = torch.mean(loss_nomean.detach(), dim=(1, 2, 3, 4))
        for i in set(task.cpu().numpy()):
            dataset_name = self.opts.adopted_datasets[i]
            log_dict[f'loss_iter/{dataset_name}'] = torch.mean(loss_diff[torch.where(task == i)])
        wandb.log(log_dict)

        loss_diff = list(loss_diff.cpu().numpy())
        dataset_diff = [self.opts.adopted_datasets[i] for i in task.cpu().numpy()]
        loss_sample = pd.DataFrame({
            'dataset': dataset_diff,
            'loss': loss_diff,
        })

        return output.cpu(), loss_sample

    def do_eval_iter(self, signal, target, task, info):

        # predict
        pred = self.predict(signal, task, self.patch_size)

        # calculate metrics
        _, stats = get_metric_stats(pred, target)

        # format outputs
        metric_sample = pd.DataFrame([stats])
        metric_sample.insert(loc=0, column='dataset', value=info['dataset'])
        metric_sample.insert(loc=1, column='path_czi', value=info['path_czi'])

        return pred, metric_sample

    def predict(self, signal, task, patch_size):
        signal = signal.to(self.device)
        task = task.to(self.device)

        self.net.eval()

        img_size = signal.shape[-3:]
        over_lap_ratio = 1 / 2
        strides = [
            int(ceil(patch_len * (1 - over_lap_ratio)))
            for patch_len in patch_size
        ]
        steps = [
            int(ceil((img_len - patch_len) / stride + 1))
            for img_len, patch_len, stride in zip(img_size, patch_size, strides)
        ]

        gauss_map = get_gaussian(patch_size)
        gauss_map = torch.from_numpy(gauss_map).to(self.device)
        pred_sum = torch.zeros(signal.shape).to(self.device)
        weight_sum = torch.zeros(signal.shape).to(self.device)

        # obtain patchs of signal
        patchs = list()
        for i in range(steps[0]):
            for j in range(steps[1]):
                for k in range(steps[2]):
                    indexs = [i, j, k]
                    starts = [
                        int(idx * stride)
                        for idx, stride in zip(indexs, strides)
                    ]
                    ends = [  # prevent overflow
                        min(start + patch_len, img_len)
                        for start, patch_len, img_len in zip(starts, patch_size, img_size)
                    ]
                    starts = [  # readjust starts
                        max(int(end - patch_len), 0)
                        for end, patch_len in zip(ends, patch_size)
                    ]
                    patchs.append({
                        'starts': starts,
                        'ends': ends,
                        'cropped': signal[:, :, starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]],
                    })

        batch_buffer = list()
        while True:
            batch_buffer.append(patchs.pop())
            if len(batch_buffer) == self.opts.batch_size_eval or len(patchs) == 0:
                # bulid batch input
                signal_patchs = torch.cat([element['cropped'] for element in batch_buffer], dim=0)
                # predict
                with torch.no_grad():
                    pred_patchs = self.net(signal_patchs, task.expand(len(batch_buffer)))
                    if isinstance(pred_patchs, tuple):
                        pred_patchs = pred_patchs[0]
                # consolidate
                for i, element in enumerate(batch_buffer):
                    starts = element['starts']
                    ends = element['ends']
                    pred_patch = pred_patchs[i].unsqueeze(0)
                    pred_sum[:, :, starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]] += \
                        pred_patch[:, :] * gauss_map
                    weight_sum[:, :, starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]] += \
                        gauss_map
                # clear buffer
                batch_buffer.clear()
                if len(patchs) == 0:
                    break

        # get final prediction
        pred = pred_sum / weight_sum

        return pred.cpu()

    def __str__(self):
        out_str = \
            """
            Network:
            {:s}
            Loss:
            {:s}
            Optimizer:
            {:s}
            """.format(
                self.nn_module.__str__(),
                self.criterion.__str__(),
                self.optimizer.__str__(),
            )
        return out_str


def get_gaussian(patch_size, sigma_scale=1 / 8):
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gauss_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gauss_map = gauss_map / np.max(gauss_map) * 1
    gauss_map = gauss_map.astype(np.float32)
    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gauss_map[gauss_map == 0] = np.min(gauss_map[gauss_map != 0])
    return gauss_map


def _set_gpu_recursive(var, gpu_id):
    """Moves Tensors nested in dict var to gpu_id.

    Modified from pytorch_integrated_cell.

    Parameters:
    var - (dict) keys are either Tensors or dicts
    gpu_id - (int) GPU onto which to move the Tensors
    """
    for key in var:
        if isinstance(var[key], dict):
            _set_gpu_recursive(var[key], gpu_id)
        elif torch.is_tensor(var[key]):
            if gpu_id == -1:
                var[key] = var[key].cpu()
            else:
                var[key] = var[key].cuda(gpu_id)
