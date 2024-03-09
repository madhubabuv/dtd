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

def viz(left_image, disp):
    disp = disp[0].detach().cpu().numpy()
    left_image = left_image.squeeze().detach().permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(left_image)
    ax[1].imshow(disp.squeeze(), cmap="plasma")
    ax[0].axis("off")
    ax[1].axis("off")
    plt.tight_layout()
    plt.savefig(viz_dir+"/disparity.png")
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
    plt.savefig(viz_dir+"/mask.png")
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
    plt.savefig(viz_dir+"/photometric_error.png")
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
        reference_idx = 0
        reference_key = "frame{}".format(reference_idx)
        left_image = data[reference_key]["image"].cuda()
        right_image = data[reference_key]["stereo_pair"].cuda()
        K = data[reference_key]["camera_matrix"].cuda()
        predicted_disparities, nn_distances, mask = depth_net(
            left_image, right_image, return_distance=True, masks=True, norm=True
        )
        pose = get_stereo_pose(left_image.shape[0])
        total_loss = 0

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

            ############ Gradient Smoothing ###############
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
                epoch, args.num_epochs, loss.item()
            )
        )
        if iter % 100 == 0:
            viz(warped_right_image[0:1], disp[:, 0:1])
            viz_error(left_image[0:1], photo_loss[0:1])
            viz_mask(warped_right_image[0:1], mask[0][0:1])

        torch.cuda.empty_cache()


if __name__ == "__main__":

    #args = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description="Depth Estimation")
    parser.add_argument("--image_height", type=int, default=192,help="image height")
    parser.add_argument("--image_width", type=int, default=320,help="image width")
    parser.add_argument("--batch_size", type=int, default=1,help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,help="learning rate")
    parser.add_argument("--dataset", type=str, default="robotcar",help="dataset")
    parser.add_argument('--data_path', type=str, 
                            default="/hdd1/madhu/data/robotcar/2014-12-16-18-44-24/stereo/",
                            help="path to the dataset")
    parser.add_argument('--ms2_train_file', type=str,
                            default="/hdd1/madhu/data/ms2/train_nighttime_list.txt",
                            help="path to the ms2 trian file, only used when dataset is ms2")
    parser.add_argument('--use_full_res',action='store_true',help="use full resolution images")
    parser.add_argument('--checkpoint_dir', type=str, 
                        default="checkpoints/test_model",
                            help="path to the checkpoint directory")
    parser.add_argument('--resume',action='store_true',help="resume training")
    parser.add_argument('--pretrained_ckpt', type=str, 
                        default="/mnt/nas/madhu/data/checkpoints/chapter_3/dino_unimatch_v2_smooth_0.1/depth_net_20.pth",
                            help="path to the pretrained checkpoint")
    parser.add_argument('--num_epochs', type=int, default=20,help="number of epochs")

    
    args = parser.parse_args()
    args.working_resolution = (args.image_width, args.image_height)
    if args.dataset == "robotcar":
        from datasets.robotcar.dataloader import RobotCarDataset
        dataset = RobotCarDataset(args)

    elif args.dataset == "ms2":
        from datasets.ms2.depth_test_dataloader import MS2Dataset
        args.test_file_path = args.ms2_train_file
        #args.data_path = "/hdd1/madhu/data/ms2"
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

    depth_net = DepthNetwork(args)
    depth_net.cuda()
    depth_net.train()

    warper = ImageWarping(args.batch_size, 
                          args.image_height, 
                          args.image_width)
    warper.cuda()

    loss_fn = PhotometricLoss()

    optimizer = torch.optim.Adam(depth_net.model.parameters(), lr=args.learning_rate)
    #checkpoint_dir = "/mnt/nas/madhu/data/checkpoints/chapter_3/dino_unimatch_v2_full"
    #checkpoint_path = "/mnt/nas/madhu/data/checkpoints/chapter_3/dino_unimatch_v2_smooth_0.1/depth_net_20.pth"
    depth_net.load_state_dict(torch.load(args.pretrained_ckpt), strict=False)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    #for visualization
    viz_dir = os.path.join(args.checkpoint_dir, "viz")
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
        
    # training loop
    for epoch in range(args.num_epochs):
        try:
            train(epoch)
            if epoch % 5 == 0:

                torch.save(
                    depth_net.state_dict(),
                    os.path.join(args.checkpoint_dir, "depth_net_{}.pth".format(epoch)),
                )

        except Exception as e:
            print(e)
            breakpoint()
