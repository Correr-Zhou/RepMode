import argparse
import os
import numpy as np
import pandas as pd


def int_or_float(x):
    try:
        val = int(x)
        assert val >= 0
    except ValueError:
        val = float(x)
        assert 0.0 <= val <= 1.0
    return val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src_csv', help='path to dataset CSV')
    parser.add_argument('dst_dir', help='destination directory of dataset split')
    parser.add_argument('--train_size', type=int_or_float, default=0.8, help='training set size as int or faction of total dataset size')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--no_shuffle', action='store_true', help='random seed')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
    opts = parser.parse_args()
    vprint = print if opts.verbose else lambda *a, **kw: None

    name = os.path.basename(opts.src_csv).split('.')[0]
    path_store_split = os.path.join(opts.dst_dir, name)
    path_train_csv = os.path.join(path_store_split, 'train.csv')
    path_test_csv = os.path.join(path_store_split, 'test.csv')
    if os.path.exists(path_train_csv) and os.path.exists(path_test_csv):
        vprint('Using existing train/test split.')
        return
    rng = np.random.RandomState(opts.seed)
    df_all = pd.read_csv(opts.src_csv)
    if not opts.no_shuffle:
        df_all = df_all.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    if opts.train_size == 0:
        df_test = df_all
        df_train = df_all[0:0]  # empty DataFrame but with columns intact
    else:
        if isinstance(opts.train_size, int):
            idx_split = opts.train_size
        elif isinstance(opts.train_size, float):
            idx_split = round(len(df_all)*opts.train_size)
        else:
            raise AttributeError
    df_train = df_all[:idx_split]
    df_test = df_all[idx_split:]
    vprint('train/test sizes: {:d}/{:d}'.format(len(df_train), len(df_test)))
    if not os.path.exists(path_store_split):
        os.makedirs(path_store_split)
    df_train.to_csv(path_train_csv, index=False)
    df_test.to_csv(path_test_csv, index=False)
    vprint('saved:', path_train_csv)
    vprint('saved:', path_test_csv)


if __name__ == '__main__':
    main()
