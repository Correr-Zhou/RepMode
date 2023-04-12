import numpy as np
import scipy.stats as stats
import sklearn.metrics as metrics
import torch


def get_metric_stats(pred, target):

    pred = pred.unsqueeze(0).unsqueeze(0).numpy()
    target = target.unsqueeze(0).unsqueeze(0).numpy()

    target_flat = target.flatten()
    pred_flat = pred.flatten()

    # error map
    err_map = np.abs(pred - target)

    # MSE
    MSE = metrics.mean_squared_error(target_flat, pred_flat)

    # MAE
    MAE = metrics.mean_absolute_error(target_flat, pred_flat)

    # R2
    R2 = metrics.r2_score(target_flat, pred_flat)


    all_stats = {
        'MSE': MSE,
        'MAE': MAE,
        'R2': R2,
    }

    return err_map, all_stats
