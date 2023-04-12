from multiprocessing.spawn import old_main_modules
from os import pathconf_names
import torch.utils.data
from fnet.data.czireader import CziReader
from fnet.data.fnetdataset import FnetDataset
import pandas as pd
import numpy as np
import pdb
import fnet.transforms
import os
from tqdm import tqdm
import torch.nn.functional as F


class SSPDataset(FnetDataset):
    def __init__(self, opts, logger, ds_type):

        # define paras
        self.opts = opts
        self.logger = logger
        self.ds_type = ds_type
        self.transform_source = self.transform_target = [
            fnet.transforms.normalize,
            fnet.transforms.Resizer((1, 0.37241, 0.37241))  # 0.108 um/px -> 0.29 um/px
        ]
        self.patch_size = (32, 128, 128)
        self.random_flip_prob = 0.5
        self.df = None
        self.data = list()

        # load entire dataset
        if self.opts.path_load_dataset is not None:
            path_load_pth = os.path.join(self.opts.path_load_dataset, self.ds_type + '.pth')
            if os.path.exists(path_load_pth):  # load saved dataset if used
                entire_ds = torch.load(path_load_pth)
                if len(opts.adopted_datasets) > 1:
                    opts.adopted_datasets = entire_ds['adopted_datasets_loaded']  # re-assgin adopted_datasets
                else:  # for Multi-Net only
                    entire_ds = self.fliter_one_cat_data(entire_ds, opts.adopted_datasets)
                self.df = entire_ds['df']
                self.data = entire_ds['data']
                self.logger.info(f'[DATASET] Dataset ({ds_type}) loaded from: {path_load_pth}')
            return

        # load with CziReader
        csvs = list()
        for ds_name in self.opts.adopted_datasets:
            path_csv = os.path.join(self.opts.path_dataset_csv, ds_name, self.ds_type + '.csv')
            single_csv = pd.read_csv(path_csv)
            single_csv.insert(loc=0, column='dataset', value=ds_name)
            csvs.append(single_csv)
        self.df = pd.concat(csvs)
        assert all(i in self.df.columns for i in ['path_czi', 'channel_signal', 'channel_target'])

        desc = '{:<10}{:<10}'.format('[N/A]', 'Loading')
        for index in tqdm(range(len(self.df)), desc=desc):

            # read czi
            element = self.df.iloc[index, :]
            has_target = not np.isnan(element['channel_target'])
            path_czi = opts.path_dataset_czi + element['path_czi'].lstrip('data')
            czi = CziReader(path_czi)

            # extract
            im_out = list()
            im_out.append(czi.get_volume(element['channel_signal']))
            if has_target:
                im_out.append(czi.get_volume(element['channel_target']))

            # transform
            if self.transform_source is not None:
                for t in self.transform_source:
                    im_out[0] = t(im_out[0])
            if has_target and self.transform_target is not None:
                for t in self.transform_target:
                    im_out[1] = t(im_out[1])
            im_out = [torch.from_numpy(im.astype(float)).float() for im in im_out]
            # unsqueeze to make the first dimension be the channel dimension
            im_out = [torch.unsqueeze(im, 0) for im in im_out]

            # format data
            self.data.append({
                'info': self.get_information(index),
                'imgs': im_out,
            })

        self.logger.info(f'[DATASET] Dataset ({ds_type}) loaded with CziReader.')

        # save data if the path exists
        if self.opts.path_save_dataset is not None:
            if not os.path.exists(self.opts.path_save_dataset):
                os.makedirs(self.opts.path_save_dataset)
            path_save_pth = os.path.join(self.opts.path_save_dataset, self.ds_type + '.pth')
            entire_ds = {
                'adopted_datasets_loaded': self.opts.adopted_datasets,
                'df': self.df,
                'data': self.data,
            }
            torch.save(entire_ds, path_save_pth)
            self.logger.info(f'[DATASET] Dataset ({ds_type}) save to: {path_save_pth}')

    def fliter_one_cat_data(self, entire_ds, adopted_datasets):
        ds_name = adopted_datasets[0]
        df = entire_ds['df']
        new_df = df[df['dataset'] == ds_name]
        entire_ds['df'] = new_df
        data = entire_ds['data']
        new_data = list()
        for i in range(len(data)):
            if data[i]['info']['dataset'] == ds_name:
                new_data.append(data[i])
        entire_ds['data'] = new_data
        self.logger.info(f'[DATASET] Fliter data: {ds_name}')
        return entire_ds

    def __getitem__(self, index):
        signal = self.data[index]['imgs'][0]
        target = self.data[index]['imgs'][1]
        info = self.data[index]['info']

        # data augmentation
        if self.ds_type == 'train':
            signal, target = self.data_aug(signal, target)

        # obtain task index
        dataset_name = info['dataset']
        task = self.opts.adopted_datasets.index(dataset_name)

        return signal, target, task

    def get_information(self, index: int) -> dict:
        return self.df.iloc[index, :].to_dict()

    def __len__(self):
        return len(self.df)

    def data_aug(self, signal, target):
        # random crop
        assert signal.shape == target.shape
        img_size = signal.shape[-3:]
        starts = np.array([
            np.random.randint(0, i - c + 1)
            for i, c in zip(img_size, self.patch_size)
        ])
        ends = starts + self.patch_size
        signal = signal[:, starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]
        target = target[:, starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]

        # random flip
        random_p = np.random.uniform(0, 1, size=3)
        filp_dims = list(np.where(random_p <= self.random_flip_prob)[0] + 1)
        signal = torch.flip(signal, dims=filp_dims)
        target = torch.flip(target, dims=filp_dims)

        return signal, target
