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
from Neighbor2Neighbor.arch_unet import UNet



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

def crop_images(image):

    shape = image.shape
    half_height = int(shape[2]/2)
    image = image[:,:,half_height:,:]
    return image

def patchify_images(image):

    batch_size = image.shape[0]

    # Specify the patch size
    #patch_size = (1, 3, 192, 320)  # 4 non-overlapping patches
    patch_size = (1, 3, 96, 160)  # 16 non-overlapping patches

    # Use unfold to extract patches
    patches = image.unfold(2, patch_size[2], patch_size[2]).unfold(3, patch_size[3], patch_size[3])

    # Reshape the patches tensor to have a shape of (1, 3, 4, 192, 320)
    patches = patches.contiguous().view(batch_size, 3, -1, patch_size[2], patch_size[3])
    patches = patches.permute(0, 2, 1, 3, 4)
    patches = patches.contiguous().view(-1, 3, patch_size[2], patch_size[3])


    # patches now contains the non-overlapping patches
    return patches

def train(epoch):

    depth_net.model.train()
    progress_bar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
    # for _,data in progress_bar: break
    # for iter in tqdm.tqdm(range(1000), total = 1000):
    for iter, data in progress_bar:
        reference_idx = 0
        reference_key = "frame{}".format(reference_idx)
        left_image = data[reference_key]["image"].cuda()
        right_image = data[reference_key]["stereo_pair"].cuda()

        day_left_image = data["frame1"]["image"].cuda()
        day_right_image = data["frame1"]["stereo_pair"].cuda()

        left_image = torch.cat([left_image, day_left_image], dim=0)
        right_image = torch.cat([right_image, day_right_image], dim=0)

        patched_left_image = patchify_images(left_image)
        patched_right_image = patchify_images(right_image)

        # left = patched_left_image.view(2, -1, 3, 192, 320)
        # left = left.view(2, 2, 2, 3, 192, 320).permute(0, 3, 1,4, 2, 5).contiguous().view(2, 3, 384, 640)
        # fig, ax = plt.subplots(2, 1, figsize=(10, 5))
        # for i in range(2):
            
        #     ax[i].imshow(left[i].permute(1, 2, 0).cpu().numpy())
        #     ax[i].axis("off")
        #     ax[i].set_title("Patch {}".format(i * 2 ))
        # plt.tight_layout()

        # plt.savefig("patched_left_image.png")

        # breakpoint()


        
        night_k = data[reference_key]["camera_matrix"].cuda()
        day_k = data["frame1"]["camera_matrix"].cuda()
        K = torch.cat([night_k, day_k], dim=0)
        pose = get_stereo_pose(left_image.shape[0])

        #crop the images
        #left_image = crop_images(left_image)
        #right_image = crop_images(right_image)

        outputs = depth_net(patched_left_image, patched_right_image, return_dict=True)
        predicted_disparities = outputs["flow_preds"]

        #breakpoint()
        nn_distances = outputs["nn_distance"]
        mask = outputs["bad_pixel_mask"]
        #breakpoint()
        total_loss = 0
        ############ koleo reg_loss ############
        weights = [0.01,1.0]
        #for idx,distance in enumerate(nn_distances):
        distance = nn_distances[-1]
        distance_loss = -1.0 * ((1 - distance) ** 2) * torch.log(distance + 1e-6)
        distance_loss = distance_loss.mean()
        total_loss += distance_loss#* weights[idx]

        # #denoise the images
        # with torch.no_grad():
        #     denoise_night_left_image = denoiser(data[reference_key]["image"].cuda())
        #     denoised_night_right_image = denoiser(data[reference_key]["stereo_pair"].cuda())
        #     denoised_left_images = torch.cat([denoise_night_left_image, day_left_image], dim=0)
        #     denoised_right_images = torch.cat([denoised_night_right_image, day_right_image], dim=0)

        # weights = [1/8,1/8,1/4,1/4,1/4]
        for idx, disp in enumerate(predicted_disparities):

            disp = disp.unsqueeze(1)
            patch_shape = disp.shape[2:]
            disp = disp.view(args.batch_size*2, 4, 4, 1, patch_shape[0], patch_shape[1]).permute(0, 3, 1, 4, 2, 5).contiguous().view(args.batch_size*2, 1, args.image_height, args.image_width)
            
            depth = (0.239983 * 100.0) / (disp + 1e-6)
            warped_right_image = warper(right_image, depth, pose, K)
            photo_loss = loss_fn.simple_photometric_loss(left_image, warped_right_image)

            # warped_right_image = warper(denoised_right_images, depth, pose, K)
            # photo_loss = loss_fn.simple_photometric_loss(denoised_left_images, warped_right_image)

            loss = photo_loss.mean(2, True).mean(3, True).mean()
            #loss = loss[:args.batch_size].mean() + 0.5 * loss[args.batch_size:].mean()
            total_loss += loss

            # ############ Gradient Smoothing ###############
            if idx < 2:
                weight = 1 / 8
            else:
                weight = 1 / 4
            smoothloss = get_disparity_smooth_loss(disp, left_image)
            total_loss += smoothloss * 0.1 * weight

        # ############ stop gradient based loss ############
        # all_z = outputs["before_features"]
        # all_h = outputs['after_features']

        # for all_z, all_h in zip(outputs['before_features'], outputs['after_features']):

        #     #shape = z.shape
        #     #all_z = z.permute(0, 2, 3, 1).contiguous().view(shape[0],shape[2]* shape[3], -1)
        #     #all_h = h.permute(0, 2, 3, 1).contiguous().view(shape[0],shape[2]* shape[3], -1)

        #     #all_z = torch.mean(all_z, dim=1)
        #     #all_h = torch.mean(all_h, dim=1)

        #     z1, z2 = torch.split(all_z, args.batch_size, dim=0)
        #     h1, h2 = torch.split(all_h, args.batch_size, dim=0)
        #     def D(p, z):
        #         z = z.detach()
        #         #normalization
        #         similarity = torch.nn.functional.cosine_similarity(p, z).mean()
        #         return similarity

        #     similarity_loss = -0.5 * (D(h1, z2) + D(h2, z1))
        #     total_loss += (similarity_loss * 0.5) * 0.1







        




        ######### optimal transport between day and night#####
        #features = outputs["features"][0]
        # alpha = 0.5
        #night_features = features[0 : args.batch_size]
        #day_features = features[args.batch_size :]
        # transport_outputs = ot(night_features, day_features)
        # matching_loss = transport_outputs["matching_loss"]
        # bin_loss = transport_outputs["dustbin_loss"]
        # transport_loss = alpha * matching_loss #+ (1-alpha) * bin_loss
        # total_loss += transport_loss * 0.1

        #transport_outputs = wasserstein(night_features, day_features.detach())
        #matching_loss = (torch.abs(night_features - day_features.detach())).mean()
        #matching_loss = transport_outputs["matching_loss"]
        #total_loss += matching_loss * 0.1


        # ############ Style loss ############
        # features = [outputs["features"][0]]
        # total_matching_loss = 0
        # for feat in features:
        #     night_features = feat[0 : args.batch_size]
        #     day_features = feat[args.batch_size :]

        #     night_mean = mu(night_features)
        #     night_sigma = sigma(night_features)

        #     day_mean = mu(day_features)
        #     day_sigma = sigma(day_features)
        
        #     matching_loss =torch.nn.functional.mse_loss(night_mean ,day_mean) + torch.nn.functional.mse_loss(night_sigma , day_sigma)
        #     total_matching_loss += matching_loss 

        # total_loss += total_matching_loss * 10.






        # if iter % 50 == 0:
        #     t_plan = transport_outputs["gamma"][0].detach().cpu().numpy()
        #     t_plan[-1,-1] = 0
        #     plt.imshow(t_plan, cmap="plasma")
        #     plt.savefig("t_plan.png")

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # progress_bar.set_description(
        #     "epoch: {}/{} training loss: {:.4f}".format(
        #         epoch, args.num_epochs, total_loss.item()
        #     )
        # )

        progress_bar.set_description(
            "epoch: {}/{} training loss: {:.4f} photo_loss: {:.4f} smooth_loss: {:.4f}, extra_loss: {:.4f}".format(
                epoch,
                args.num_epochs,
                total_loss.item(),
                loss.item(),
                smoothloss.item(),
                smoothloss.item()

            )
        )

        if iter % 50 == 0:
            viz(warped_right_image[0:1], disp[:, 0:1], "night_disp")
            viz_error(left_image[0:1], photo_loss[0:1], "night_photo")
            viz_mask(warped_right_image[0:1], mask[0][0:1], "night_mask")

            viz(
                warped_right_image[args.batch_size : args.batch_size + 1],
                disp[args.batch_size : args.batch_size + 1, 0:1],
                "day_disp",
            )
            viz_error(
                left_image[args.batch_size : args.batch_size + 1],
                photo_loss[args.batch_size : args.batch_size + 1],
                "day_photo",
            )
            viz_mask(
                warped_right_image[args.batch_size : args.batch_size + 1],
                mask[0][args.batch_size : args.batch_size + 1],
                "day_mask",
            )
            
            

        torch.cuda.empty_cache()


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    # args = get_test_args()
    args.image_height = 384
    args.image_width = 640
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

    args.dataset = "robotcar"  # robotcar, ms2

    if args.dataset == "robotcar":
        # from datasets.robotcar.dataloader import RobotCarDataset
        from datasets.robotcar.day_night_paried_dataset import (
            DayNightDataset as RobotCarDataset,
        )

        args.data_path = "/hdd1/madhu/data/robotcar"  # /2014-12-16-18-44-24/stereo/"
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
        num_workers=8,
        pin_memory=True,
        sampler=None,
    )

    depth_net = DepthNetwork(args, reg_refine=False)
    depth_net.cuda()
    depth_net.train()

    warper = ImageWarping(args.batch_size * 2, args.image_height, args.image_width)
    warper.cuda()

    loss_fn = PhotometricLoss()

    # ot = OT()
    # ot.cuda()
    #params = list(depth_net.model.parameters())+list(ot.parameters())

    #denoiser
    args.n_channel = 3 
    args.n_feature = 48
    # denoiser = UNet(in_nc=args.n_channel,
    #            out_nc=args.n_channel,
    #            n_feature=args.n_feature)
    # denoiser.cuda()
    # denoiser.eval()
    # denoiser_checkpoint = '/code/deep-slam/packages/Neighbor2Neighbor/pretrained_model/model_gauss5-50_b4e100r02.pth'
    # if os.path.isfile(denoiser_checkpoint):
    #     denoiser.load_state_dict(torch.load(denoiser_checkpoint))
    #     print('Denoiser Loaded')
    # else:
    #     print('Denoiser Checkpoint not found')    

    #############################################

    optimizer = torch.optim.Adam(depth_net.model.parameters(), lr=args.learning_rate)
    checkpoint_dir = (
        "/mnt/nas/madhu/data/checkpoints/chapter_4_cvpr/dino_v1_d_n_half_images"
    )
    checkpoint_path = (
        "/mnt/nas/madhu/data/checkpoints/chapter_3/dino_unimatch_v1/depth_net_10.pth"
    )
    depth_net.load_state_dict(torch.load(checkpoint_path), strict=False)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # training loop
    args.num_epochs = 21
    #train(0)

    ### ALWAYS check the camera calibration matrices and scale before trianing


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
