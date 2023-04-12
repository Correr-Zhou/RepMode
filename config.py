import argparse


def get_arguments_main():

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
    parser.add_argument('--nn_module', default='RepMode', help='name of the model')
    parser.add_argument('--num_epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='size of each batch')
    parser.add_argument('--batch_size_eval', type=int, default=8, help='size of each batch for evaluation')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')

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
    parser.add_argument('--debugging', action='store_true', help='set to debug')
    parser.add_argument('--save_test_preds', action='store_true', help='set to save predicted results in test')
    parser.add_argument('--save_test_signals_and_targets', action='store_true', help='set to save signals and targets in test')
    parser.add_argument('--monitor_model', action='store_true', help='set to monitor model')

    # checkpoint
    parser.add_argument('--epoch_checkpoint', nargs='+', type=int, default=[], help='epochs at which to save checkpoints of the model')
    parser.add_argument('--interval_checkpoint', type=int, help='interval of epochs for saving checkpoints')

    # val
    parser.add_argument('--interval_val', type=int, default=20, help='interval of epochs for performing validation')

    # log
    parser.add_argument(
        '--run_name',
        default="f'[{opts.exp_name}] [{opts.nn_module}]'",
        type=str,
        help='run name for logging'
    )
    parser.add_argument(
        '--tags',
        nargs='+',
        type=str,
        help='tags for logging'
    )
    parser.add_argument(
        '--id',
        type=str,
        help='id for logging'
    )

    return parser.parse_args()
