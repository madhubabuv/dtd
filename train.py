import torch
import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from models.depth_model import StereoDepthNet as DepthNetwork
from models.image_warping import ImageWarping
from losses.photometric_loss import PhotometricLoss
from losses.regularization_loss import get_disparity_smooth_loss
import argparse


def test():
    depth_net.eval()
    for idx, data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        reference_idx = 0
        reference_key = "frame{}".format(reference_idx)
        left_image = data[reference_key]["image"].cuda()
        right_image = data[reference_key]["stereo_pair"].cuda()
        with torch.no_grad():
            disp = depth_net(left_image, right_image)
        viz(left_image, disp)


def viz(left_image, disp):
    disp = disp[0].detach().cpu().numpy()
    left_image = left_image.squeeze().detach().permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(left_image)
    ax[1].imshow(disp.squeeze(), cmap="plasma")
    ax[0].axis("off")
    ax[1].axis("off")
    plt.tight_layout()
    plt.savefig("test.png")
    plt.clf()
    plt.close()


def viz_mask(left_image, disp):
    disp = disp[0].detach().cpu().numpy()
    left_image = left_image.squeeze().detach().permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(left_image)
    ax[1].imshow(disp.squeeze(), cmap="plasma")
    ax[0].axis("off")
    ax[1].axis("off")
    plt.tight_layout()
    plt.savefig("mask.png")
    plt.clf()
    plt.close()


def viz_error(image, error):
    image = image.squeeze().detach().permute(1, 2, 0).cpu().numpy()
    error = error.squeeze().detach().cpu().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(image)
    ax[1].imshow(error.squeeze())
    ax[0].axis("off")
    plt.tight_layout()
    plt.savefig("photo_error.png")
    plt.clf()
    plt.close()


def get_stereo_pose(batch_size):
    # assuming the dataset is KITTI
    LEFT_TO_RIGHT_STEREO_POSE = np.eye(4, dtype=np.float32)
    LEFT_TO_RIGHT_STEREO_POSE[0, 3] = -1  # 0.239983 # baseline
    LEFT_TO_RIGHT_STEREO_POSE = torch.from_numpy(LEFT_TO_RIGHT_STEREO_POSE).cuda()
    LEFT_TO_RIGHT_STEREO_POSE = LEFT_TO_RIGHT_STEREO_POSE.unsqueeze(0).repeat(
        batch_size, 1, 1
    )
    return LEFT_TO_RIGHT_STEREO_POSE


def train(epoch):

    depth_net.model.train()
    
    progress_bar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))

    for iter, data in progress_bar:
        optimizer.zero_grad()
        reference_idx = 0
        reference_key = "frame{}".format(reference_idx)
        left_image = data[reference_key]["image"].cuda()
        right_image = data[reference_key]["stereo_pair"].cuda()
        K = data[reference_key]["camera_matrix"].cuda()
        pose = get_stereo_pose(left_image.shape[0])
        total_loss = torch.zeros(1).cuda()

        if mixed_precision:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                predicted_disparities, nn_distances, mask = depth_net(
                    left_image, right_image, norm = False,
                )
        else:
            predicted_disparities, nn_distances, mask = depth_net(
                left_image, right_image, norm = False,
            )

        if args.distance_reg:
            ############ koleo reg_loss ############
            distance = nn_distances[-1]
            distance_loss = -1.0 * ((1 - distance) ** 2) * torch.log(distance + 1e-6)
            distance_loss = distance_loss.mean()
            total_loss += distance_loss

        for idx, disp in enumerate(predicted_disparities):
            disp = disp.unsqueeze(1)
            depth = 100.0 / (disp + 1e-6)
            warped_right_image = warper(right_image, depth, pose, K)
            photo_loss = loss_fn.simple_photometric_loss(left_image, warped_right_image)
            loss = photo_loss.mean(2, True).mean(3, True).mean()
            total_loss += loss

            if args.disp_smooth:
                ############ Gradient Smoothing ###############
                if idx < 2:
                    weight = 1 / 8
                else:
                    weight = 1 / 4
                smoothloss = get_disparity_smooth_loss(disp, left_image)
                total_loss += smoothloss * 0.1 * weight

        if mixed_precision:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(depth_net.model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(depth_net.model.parameters(), 10.0)
            optimizer.step()

        progress_bar.set_description(
            "epoch: {}/{} training loss: {:.4f}".format(
                epoch, args.num_epochs, loss.item()
            )
        )
        if iter % 50 == 0:
            viz(warped_right_image[0:1], disp[:, 0:1])
            viz_error(left_image[0:1], photo_loss[0:1])
            if args.feature_mask:
                viz_mask(warped_right_image[0:1], mask[0][0:1])

        torch.cuda.empty_cache()


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    # args = get_test_args()
    args.image_height = 192
    args.image_width = 320
    args.working_resolution = (args.image_width, args.image_height)
    args.use_gt_poses = False
    args.use_gray_scale = False
    args.use_stereo = True
    args.seq_length = 1
    args.stride = 1
    args.use_multi_scale_images = False
    args.undistort = False
    args.use_full_res = False
    args.use_seq = False
    args.use_pose = False
    args.batch_size = 8
    args.split = "train"
    args.learning_rate = 1e-4

    args.dataset = "robotcar" # robotcar, ms2
    if args.dataset == "robotcar":
        from datasets.robotcar.dataloader import RobotCarDataset
        #args.data_path = "/hdd1/madhu/data/robotcar/2014-12-16-18-44-24/stereo/"
        args.data_path = "/hdd1/madhu/data/robotcar/2014-12-09-13-21-02/stereo/"
        dataset = RobotCarDataset(args)

    elif args.dataset == "ms2":
        from datasets.ms2.depth_test_dataloader import MS2Dataset
        args.test_file_path = '/hdd1/madhu/data/ms2/train_nighttime_list.txt'
        args.data_path = "/hdd1/madhu/data/ms2"
        dataset = MS2Dataset(args)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        sampler=None,
    )


    args.feature_mask = False
    args.distance_reg = False
    args.disp_smooth = False
    args.reg_refine = False
    mixed_precision = False

    depth_net = DepthNetwork(args, feature_mask = args.feature_mask, reg_refine=args.reg_refine)
    depth_net.cuda()
    depth_net.train()

    warper = ImageWarping(args.batch_size, args.image_height, args.image_width)
    warper.cuda()

    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    loss_fn = PhotometricLoss()

    optimizer = torch.optim.Adam(depth_net.model.parameters(), lr=args.learning_rate)
    checkpoint_dir = "/mnt/nas/madhu/data/checkpoints/chapter_3/dino_unimatch_v1_day"
    #checkpoint_path = "/mnt/nas/madhu/data/checkpoints/chapter_3/dino_unimatch_v2_smooth_0.1/depth_net_20.pth"
    #depth_net.load_state_dict(torch.load(checkpoint_path), strict=False)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # training loop
    args.num_epochs = 50
    train(0)
    for epoch in range(args.num_epochs):
        try:
            train(epoch)
            if epoch % 5 == 0:

                torch.save(
                    depth_net.state_dict(),
                    os.path.join(checkpoint_dir, "depth_net_{}.pth".format(epoch)),
                )

        except Exception as e:
            print(e)
            breakpoint()
