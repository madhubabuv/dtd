from __future__ import absolute_import, division, print_function

import os
import tqdm
import cv2
import numpy as np
import pdb

cv2.setNumThreads(0)  

#KITTI_SCALE_FACTOR = 5.4
#ROBOTCAR_SCALE_FACTOR = 0.239983 * 983.044006
#MS2_SCALE_FACTOR = (0.2991842 * 764.51385)
#MS2_SCALE_FACTOR = (0.2991842 * 100)
#ROBOTCAR_SCALE_FACTOR = 0.239983*100.* 0.2399
ROBOTCAR_SCALE_FACTOR = 0.239983*983.044006 * 0.25
#MS2_SCALE_FACTOR = 0.2991842 * 100.


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
    eval_split = "normal"
    print("-> Evaluating")

    errors = []
    all_ratios = []
    for i in tqdm.tqdm(range(pred_disps.shape[0]),total = pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
    
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = ROBOTCAR_SCALE_FACTOR/ (pred_disp + 1e-6)

        if eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        if mask.sum() == 0:
            print("mask sum is 0")
            continue

        # ratio = np.median(gt_depth[mask]) / np.median(pred_depth[mask])
        # pred_depth *= ratio
        # all_ratios.append(ratio)

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        errors.append(compute_errors(gt_depth, pred_depth))


    mean_errors = np.array(errors).mean(0)
    #breakpoint()
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

    #gt_depths = np.load(args.gt_depths_path, fix_imports=True, encoding='latin1',allow_pickle=True)["data"]

    
    results  = evaluate(pred_disps, gt_depths)

    print(results)