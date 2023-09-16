from __future__ import absolute_import, division, print_function

import os
import tqdm
import cv2
import numpy as np
import pdb

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

#ROBOTCAR_SCALE_FACTOR = (0.239983 * 983.044006)
MS2_SCALE_FACTOR = (0.2991842 * 764.51385)
#ROBOTCAR_SCALE_FACTOR = 0.23998*100.
#MS2_SCALE_FACTOR = (0.2991842 * 100)


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


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


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
        #pred_depth = ROBOTCAR_SCALE_FACTOR / (pred_disp + 1e-6)
        pred_depth = MS2_SCALE_FACTOR / (pred_disp + 1e-6)
        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        # mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        # crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
        #                     0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
        # crop_mask = np.zeros(mask.shape)
        # crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        # mask = np.logical_and(mask, crop_mask)
        if mask.sum() == 0:
            print("mask sum is 0")
            continue

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        bin_inds = np.digitize(gt_depth,histogram_bins)
        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        bin_wise_metrics = []
        #if len(np.unique(bin_inds)) < 10: continue
        for bin_ind in range(len(histogram_bins)):
            bin_mask = bin_inds == bin_ind
            bin_gt_depth = gt_depth[bin_mask]
            bin_pred_depth = pred_depth[bin_mask]
            if sum(bin_mask) == 0:
                bin_errors = np.array([999,999,999,999,999,999,999])
            else:
                bin_errors = compute_errors(bin_gt_depth,bin_pred_depth)
            bin_wise_metrics.append(bin_errors)

        all_bin_wise_metrics.append(np.array(bin_wise_metrics))
        errors.append(compute_errors(gt_depth, pred_depth))


    mean_errors = np.array(errors).mean(0)
    all_bin_wise_metrics = np.stack(all_bin_wise_metrics)
    np.save("bin_wise_metrics_MS2_sgm.npy",all_bin_wise_metrics)
    return mean_errors

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_disp_path", type=str, required=True, help="path to the disparities to evaluate")
    parser.add_argument("--gt_depths_path", type=str, required=True, help="path to the ground truth depths")
    #parser.add_argument("--eval_split", type=str,  choices=["eigen", "eigen_benchmark", "garg"])
    #parser.add_argument("--disable_median_scaling", action="store_true", help="if set, don't scale the disparities by the median ratio")

    args = parser.parse_args()
    print("-> Loading predictions from ",args.pred_disp_path)
    pred_disps = np.load(args.pred_disp_path)
    print("-> Loading ground truth from ",args.gt_depths_path)
    gt_depths = np.load(args.gt_depths_path, allow_pickle=True)
    results  = evaluate(pred_disps, gt_depths)

    print(results)