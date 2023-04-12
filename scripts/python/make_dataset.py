import argparse
import os
from re import I
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src_dir', help='directory to used dataset CSV')
    parser.add_argument('src_csv', help='path to target dataset CSV')
    parser.add_argument('dst_dir', help='destination path to target dataset CSV')
    parser.add_argument('ds_type', help='dataset type')
    parser.add_argument('--sample_num', type=int, default=54, help='sample number')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--no_shuffle', action='store_true', help='random seed')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
    parser.add_argument(
        '--used_ds',
        nargs='+',
        default=[
            'alpha_tubulin',
            'beta_actin',
            'desmoplakin',
            'fibrillarin',
            'lamin_b1',
            'membrane_caax_63x',
            'myosin_iib',
            'sec61_beta',
            'st6gal1',
            'tom20',
            'zo1',
        ],
        help='destination path to dataset CSV'
    )
    opts = parser.parse_args()
    vprint = print if opts.verbose else lambda *a, **kw: None

    ds_name = os.path.basename(opts.src_csv).rstrip('.csv')
    path_ds_csv = os.path.join(opts.dst_dir, ds_name, opts.ds_type + '.csv')
    if os.path.exists(path_ds_csv):
        vprint('Using existing train/val split.')
        return

    rng = np.random.RandomState(opts.seed)

    csvs = list()
    for ds in opts.used_ds:
        path_csv = os.path.join(opts.src_dir, ds, opts.ds_type + '.csv')
        csv = pd.read_csv(path_csv)
        csvs.append(csv)
    df_used_ds = pd.concat(csvs)
    df_src = pd.read_csv(opts.src_csv)

    if not opts.no_shuffle:
        df_used_ds = df_used_ds.sample(frac=1.0, random_state=rng).reset_index(drop=True)

    cnt = 0
    idxs = np.arange(len(df_used_ds))
    rng.shuffle(idxs)
    selected_samples = list()
    for idx in idxs:
        if df_used_ds.iloc[idx]['path_czi'] in df_src['path_czi'].tolist():
            row = df_src[df_src.values == df_used_ds.iloc[idx]['path_czi']].index
            selected_samples.append(df_src.iloc[row])
            cnt += 1
        if cnt >= opts.sample_num:
            break

    vprint('sample num: {}'.format(cnt))

    path_store_ds = os.path.join(opts.dst_dir, ds_name)
    if not os.path.exists(path_store_ds):
        os.makedirs(path_store_ds)
    df_target = pd.concat(selected_samples)
    df_target.to_csv(path_ds_csv, index=False)
    vprint('saved:', path_ds_csv)


if __name__ == '__main__':
    main()
