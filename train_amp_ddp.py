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
from diffusers.optimization import get_scheduler
import argparse
from Neighbor2Neighbor.arch_unet import UNet
#from models.optimal_transport import PartialOT as OT
import torch.backends.cudnn as cudnn
import ot
import apex
from apex import amp, optimizers
from apex.parallel import DistributedDataParallel as DDP


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




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


def viz(left_image, disp, name):
    disp = disp[0].detach().cpu().numpy()
    left_image = left_image.squeeze().detach().permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(left_image)
    ax[1].imshow(disp.squeeze(), cmap="plasma")
    ax[0].axis("off")
    ax[1].axis("off")
    plt.tight_layout()
    plt.savefig(name + ".png")
    plt.clf()
    plt.close()


def viz_mask(left_image, disp, name):
    disp = disp[0].detach().cpu().numpy()
    left_image = left_image.squeeze().detach().permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(left_image)
    ax[1].imshow(disp.squeeze(), cmap="plasma")
    ax[0].axis("off")
    ax[1].axis("off")
    plt.tight_layout()
    plt.savefig(name + ".png")
    plt.clf()
    plt.close()


def viz_error(image, error, name):
    image = image.squeeze().detach().permute(1, 2, 0).cpu().numpy()
    error = error.squeeze().detach().cpu().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(image)
    ax[1].imshow(error.squeeze())
    ax[0].axis("off")
    plt.tight_layout()
    plt.savefig(name + ".png")
    plt.clf()
    plt.close()

def viz_std(stds, name):

    plt.plot(np.arange(len(stds)), stds)
    plt.savefig(name + ".png")
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

def wasserstein(x,y):   

    shape = x.shape
    re_x = x.permute(0,2,3,1).reshape(shape[0],-1,shape[1])
    re_y = y.permute(0,2,3,1).reshape(shape[0],-1,shape[1])
    distance = torch.cdist(re_x,re_y,p=2)

    # with torch.no_grad():
    # cost = distance.detach().cpu()
    re_u = torch.tensor([]).cuda()
    re_v = torch.tensor([]).cuda()
    w_dist = 0
    logs = []
    for idx in range(shape[0]):
        gamma = ot.emd(re_u,re_v, distance[idx], numItermax=5)
        logs.append(gamma)
    logs = torch.stack(logs).cuda()
    w_dist = torch.sum(logs * distance, dim = [1,2])
    w_dist = torch.mean(w_dist)
    outputs ={}
    outputs['matching_loss'] = w_dist
    outputs['gamma'] = logs
    return outputs

def mu(x):
    """ Takes a (n,c,h,w) tensor as input and returns the average across
    it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
    return torch.sum(x,(2,3))/(x.shape[2]*x.shape[3])

def sigma(x):
    """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
    across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
    the permutations are required for broadcasting"""
    return torch.sqrt((torch.sum((x.permute([2,3,0,1])-mu(x)).permute([2,3,0,1])**2,(2,3))+0.000000023)/(x.shape[2]*x.shape[3]))



def train(epoch):

    depth_net.model.train()
    progress_bar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
    #for _,data in progress_bar: break
    #for iter in tqdm.tqdm(range(1000), total = 1000):
    for iter, data in progress_bar:
        reference_idx = 0
        reference_key = "frame{}".format(reference_idx)
        left_image = data[reference_key]["image"].cuda()
        right_image = data[reference_key]["stereo_pair"].cuda()

        day_left_image = data["frame1"]["image"].cuda()
        day_right_image = data["frame1"]["stereo_pair"].cuda()
        left_image = torch.cat([left_image, day_left_image], dim=0)
        right_image = torch.cat([right_image, day_right_image], dim=0)

        #estimate depth
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


        total_loss = 0
        #pose = get_stereo_pose(left_image.shape[0])
        #night_k = data[reference_key]["camera_matrix"].cuda()
        #day_k = data["frame1"]["camera_matrix"].cuda()
        #K = torch.cat([night_k, day_k], dim=0)


        for idx, disp in enumerate(predicted_disparities):
            disp = disp.unsqueeze(1)

            flow = torch.cat([disp, torch.zeros_like(disp)], dim=1)
            flow = -1 * flow
            warped_right_image = flow_warp(right_image, flow)

            #depth = 100.*0.2399 / (disp + 1e-6)
            #warped_right_image = warper(right_image, depth, pose, K)

            #ddim_night_left_image = data[reference_key]['ddim_image'].cuda()
            #ddim_night_right_image = data[reference_key]['ddim_stereo_pair'].cuda()
            #left_image = torch.cat([ddim_night_left_image, day_left_image], dim=0)
            #right_image = torch.cat([ddim_night_right_image, day_right_image], dim=0)

            photo_loss = loss_fn.simple_photometric_loss(left_image, warped_right_image)
            loss = photo_loss.mean(2, True).mean(3, True).mean()
            total_loss += loss
            
            # ############ Gradient Smoothing ###############
            # if idx < 2:
            #     weight = 1 / 8
            # else:
            #     weight = 1 / 4
            # smoothloss = get_disparity_smooth_loss(disp, left_image)
            # total_loss += smoothloss * 0.1 * weight
            

        optimizer.zero_grad()
        with amp.scale_loss(total_loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        #loss.backward()
        optimizer.step()
        #lr_scheduler.step()

        progress_bar.set_description(
            "epoch: {}/{} training loss: {:.4f}  photo_loss: {:.4f}".format(
                epoch, args.num_epochs, total_loss.item(), loss.item()
            )
        )

        if iter % 50 == 0:
            #all_stds.append(np.mean(stds))
            #viz_std(all_stds, "std")
            #stds = []

            viz(warped_right_image[0:1], disp[:, 0:1], "night_disp")
            viz_error(left_image[0:1], photo_loss[0:1], "night_photo")
            viz_mask(warped_right_image[0:1], mask[0][0:1], "night_mask")

            # viz(
            #     warped_right_image[args.batch_size : args.batch_size + 1],
            #     disp[args.batch_size : args.batch_size + 1, 0:1],
            #     "day_disp",
            # )
            # viz_error(
            #     left_image[args.batch_size : args.batch_size + 1],
            #     photo_loss[args.batch_size : args.batch_size + 1],
            #     "day_photo",
            # )


if __name__ == "__main__":


    seed = 1234
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    

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
    args.batch_size = 4
    args.split = "train"
    args.learning_rate = 1e-4

    args.dataset = "robotcar"  # robotcar, ms2

    if args.dataset == "robotcar":
        from datasets.robotcar.day_night_paried_dataset import (
           DayNightDataset as RobotCarDataset,
        )
        args.data_path = "/hdd1/madhu/data/robotcar"  # /2014-12-16-18-44-24/stereo/"

        #from datasets.robotcar.dataloader import RobotCarDataset
        #args.data_path = '/hdd1/madhu/data/robotcar/2014-12-16-18-44-24/stereo'
        dataset = RobotCarDataset(args)

    elif args.dataset == "ms2":
        from datasets.ms2.depth_test_dataloader import MS2Dataset

        args.test_file_path = "/hdd1/madhu/data/ms2/train_nighttime_list.txt"
        args.data_path = "/hdd1/madhu/data/ms2"
        dataset = MS2Dataset(args)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        sampler=None,
    )

    
    depth_net = DepthNetwork(args, reg_refine=False)
    depth_net.cuda()
    depth_net.train()

    warper = ImageWarping(args.batch_size * 2, args.image_height, args.image_width)
    warper.cuda()

    loss_fn = PhotometricLoss()
    optimizer = torch.optim.Adam(depth_net.parameters(), lr=args.learning_rate)
    checkpoint_dir = (
        "/mnt/nas/madhu/data/checkpoints/chapter_4_cvpr/d_n_same_transfer_fusion_fp_16_v2_with_koleo_ablation"
    )
    checkpoint_path = "/mnt/nas/madhu/data/checkpoints/chapter_4_cvpr/d_n_same_transfer_fusion_fp_16_v2/depth_net_15.pth"
    depth_net.load_state_dict(torch.load(checkpoint_path), strict=True)
    
    depth_net, optimizer = amp.initialize(depth_net, optimizer,
                                      opt_level='O1',
                                      keep_batchnorm_fp32=None,
                                      loss_scale=None,
                                      )


    #Initialize the learning rate scheduler
    # lr_scheduler = get_scheduler(
    #     'cosine',
    #     optimizer=optimizer,
    #     num_warmup_steps=500, 
    #     num_training_steps=(len(dataloader) * 20),
    # )


    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # training loop
    args.num_epochs = 21
    for epoch in range(args.num_epochs):
        try:
            train(epoch)
            # if epoch % 5 == 0:

            torch.save(
                depth_net.state_dict(),
                os.path.join(checkpoint_dir, "depth_net_{}.pth".format(epoch)),
            )

        except Exception as e:
            print(e)
            breakpoint()
