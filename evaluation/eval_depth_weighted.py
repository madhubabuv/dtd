from __future__ import absolute_import, division, print_function
import os
import tqdm
import cv2
import numpy as np
import pdb
cv2.setNumThreads(0) 
#ROBOTCAR_SCALE_FACTOR = (0.239983 * 983.044006)
#MS2_SCALE_FACTOR = (0.2991842 * 764.51385)
ROBOTCAR_SCALE_FACTOR = 0.23998*100.
#MS2_SCALE_FACTOR = (0.2991842 * 100)

def weighted_mean(data):
    weights = data != 999
    weights = weights.astype(np.float32)
    data = weights * data
    data = np.sum(data,axis=0)
    weights = np.sum(weights,axis=0)
    data = data / weights

    return data
def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def evaluate(pred_disps, gt_depths):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 50
    histogram_bins = np.arange(0 , MAX_DEPTH+5)[::5]

    errors = []
    all_bin_wise_metrics = []
    for i in tqdm.tqdm(range(pred_disps.shape[0]),total = pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]
        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = ROBOTCAR_SCALE_FACTOR / (pred_disp + 1e-6)
        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        if mask.sum() == 0:
            print("mask sum is 0")
            continue
        pred_depth = pred_depth[mask]
        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        gt_depth = gt_depth[mask]

        bin_inds = np.digitize(gt_depth,histogram_bins)
        bin_wise_metrics = []
        sum_inds = 0
        for bin_ind in range(1,len(histogram_bins)):
            bin_mask = bin_inds == bin_ind
            sum_inds += sum(bin_mask)
            bin_gt_depth = gt_depth[bin_mask]
            bin_pred_depth = pred_depth[bin_mask]
            if sum(bin_mask) == 0:
                bin_errors = np.array([999,999,999,999,999,999,999])
            else:
                bin_errors = compute_errors(bin_gt_depth,bin_pred_depth)
            bin_wise_metrics.append(bin_errors)

        all_bin_wise_metrics.append(np.array(bin_wise_metrics))
        errors.append(compute_errors(gt_depth, pred_depth))
        assert sum_inds == mask.sum()

    mean_errors = np.array(errors).mean(0)
    all_bin_wise_metrics = np.stack(all_bin_wise_metrics)
    bin_means = weighted_mean(all_bin_wise_metrics).mean(0)
    return mean_errors, bin_means

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_disp_path", type=str, required=True, help="path to the disparities to evaluate")
    parser.add_argument("--gt_depths_path", type=str, required=True, help="path to the ground truth depths")
    args = parser.parse_args()
    print("-> Loading predictions from ",args.pred_disp_path)
    pred_disps = np.load(args.pred_disp_path)
    print("-> Loading ground truth from ",args.gt_depths_path)
    gt_depths = np.load(args.gt_depths_path, allow_pickle=True)
    unweighted_res, weighted_res  = evaluate(pred_disps, gt_depths)

    print("-> Unweighted Metrics")
    print(unweighted_res)
    print("-> Weighted Metrics")
    print(weighted_res)