import torch
import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from models.depth_model import StereoDepthNet as DepthNetwork
from models.image_warping import ImageWarping
from losses.photometric_loss import PhotometricLoss
from losses.regularization_loss import get_disparity_smooth_loss
from unimatch.unimatch.geometry import flow_warp
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


def viz_error(image, error, name=''):
    image = image.squeeze().detach().permute(1, 2, 0).cpu().numpy()
    error = error.squeeze().detach().cpu().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(image)
    ax[1].imshow(error.squeeze())
    ax[0].axis("off")
    plt.tight_layout()
    plt.savefig(name+"photo_error.png")
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
    for _,data in progress_bar: break
    for iter in tqdm.tqdm(range(1000), total = 1000):
    #for iter, data in progress_bar:
        reference_idx = 0
        reference_key = "frame{}".format(reference_idx)
        left_image = data[reference_key]["image"].cuda()
        right_image = data[reference_key]["stereo_pair"].cuda()
        K = data[reference_key]["camera_matrix"].cuda()
        # predicted_disparities, nn_distances, mask = depth_net(
        #     left_image, right_image, return_distance=True, masks=True, norm=True
        # )
        outputs = depth_net(left_image, right_image, return_dict=True)
        predicted_disparities = outputs["flow_preds"]
        nn_distances = outputs["nn_distance"]
        mask = outputs["bad_pixel_mask"]
        pose = get_stereo_pose(left_image.shape[0])
        total_loss = 0

        ############ koleo reg_loss ############
        distance = nn_distances[-1]
        distance_loss = -1.0 * ((1 - distance) ** 2) * torch.log(distance + 1e-6)
        distance_loss = distance_loss.mean()
        total_loss += distance_loss

        # weights = [1/8,1/8,1/4,1/4,1/4]
        for idx, disp in enumerate(predicted_disparities):
            disp = disp.unsqueeze(1)
            depth = (0.239983 * 100.) / (disp + 1e-6)
            #depth = 100.0 / (disp + 1e-6)
            #disp_pad = torch.cat([disp, torch.zeros_like(disp)], dim=1)
            #warped_right_image = flow_warp(right_image, -1.*disp_pad)
            #
            warped_right_image = warper(right_image, depth, pose, K)
            photo_loss = loss_fn.simple_photometric_loss(left_image, warped_right_image)
            loss = photo_loss.mean(2, True).mean(3, True).mean()
            total_loss += loss

            # #lets focus on heavy photo regions
            #focal_loss = -1.*((photo_loss.detach())**0.5)*torch.log( 1 - photo_loss)
            #total_loss += focal_loss.mean()
            #total_loss = -1 * photo_loss* torch.log(1 - photo_loss).mean()
            # loss_2 = focal_loss.mean(2, True).mean(3, True).mean()
            # total_loss += loss_2


            # ############ Gradient Smoothing ###############
            if idx < 2:
                weight = 1 / 8
            else:
                weight = 1 / 4
            smoothloss = get_disparity_smooth_loss(disp, left_image)
            total_loss += smoothloss * 0.1 * weight

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        progress_bar.set_description(
            "epoch: {}/{} training loss: {:.4f}".format(
                epoch, args.num_epochs, total_loss.item()
            )
        )
        if iter % 50 == 0:
            viz(warped_right_image[0:1], disp[:, 0:1])
            viz_error(left_image[0:1], photo_loss[0:1])
            viz_mask(warped_right_image[0:1], mask[0][0:1])
            #viz_error(left_image[0:1], focal_loss[0:1],name = 'focal')

        #torch.cuda.empty_cache()


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
    args.batch_size = 1
    args.split = "train"
    args.learning_rate = 1e-4

    args.dataset = "robotcar" # robotcar, ms2

    if args.dataset == "robotcar":
        from datasets.robotcar.dataloader import RobotCarDataset
        args.data_path = "/hdd1/madhu/data/robotcar/2014-12-16-18-44-24/stereo/"
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

    depth_net = DepthNetwork(args, reg_refine=False)
    depth_net.cuda()
    depth_net.train()

    warper = ImageWarping(args.batch_size, args.image_height, args.image_width)
    warper.cuda()

    loss_fn = PhotometricLoss()

    optimizer = torch.optim.Adam(depth_net.model.parameters(), lr=args.learning_rate)
    checkpoint_dir = "/mnt/nas/madhu/data/checkpoints/chapter_4_cvpr/focal_photometric_loss"
    checkpoint_path = "/mnt/nas/madhu/data/checkpoints/chapter_3/dino_unimatch_v1/depth_net_10.pth"
    depth_net.load_state_dict(torch.load(checkpoint_path), strict=False)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # training loop
    args.num_epochs = 20
    train(0)
    for epoch in range(args.num_epochs):
        try:
            train(epoch)
            #if epoch % 5 == 0:

            torch.save(
                depth_net.state_dict(),
                os.path.join(checkpoint_dir, "depth_net_{}.pth".format(epoch)),
            )

        except Exception as e:
            print(e)
            breakpoint()
