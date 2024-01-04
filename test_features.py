import tqdm
import numpy as np
import torch
import os
import cv2
from models.depth_model import StereoDepthNet
from matplotlib import pyplot as plt

from git_repos.day_night_diffusion.uncond_image_generation.denoise_image import get_model,denoise


def put_text(image,text):

    np_image = image * 255
    np_image = np_image.astype(np.uint8)
    np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    cv2.putText(np_image,text,(180,20), cv2.FONT_HERSHEY_DUPLEX, 0.4,(255,255,255),1, cv2.LINE_AA)

    np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    np_image = np_image.astype(np.float32) / 255
    return np_image
    
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

def plot_features(night_image, night_feat, day_image, day_feat):

    fig, ax = plt.subplots(2,2, figsize=(20,10))
    ax[0,0].imshow(night_image)
    ax[0,1].imshow(night_feat)
    ax[1,0].imshow(day_image)
    ax[1,1].imshow(day_feat)
    #ax[2].imshow(masks[0].squeeze().detach().cpu().numpy(), cmap='plasma')
    #ax[0].axis('off')
    #ax[1].axis('off')
    #ax[2].axis('off')
    plt.tight_layout()
    plt.savefig('features.png')  
    plt.clf() 


def test():
    depth_net.eval()
    depth_net.model.eval()
    predictions = []

    for idx, data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        reference_idx = 0
        reference_key = "frame{}".format(reference_idx)

        if data[reference_key]['timestamp'][0] != '1418756885462989':
            continue


        left_image = data[reference_key]["image"].cuda()
        right_image = data[reference_key]["stereo_pair"].cuda()



        # left_image = left_image ** 0.45
        # right_image = right_image**0.45

        pair_image = data[reference_key]['pair_image'].cuda()
        pair_stereo_pair = data[reference_key]['pair_stereo_pair'].cuda()
        
        left_image = torch.nn.functional.interpolate(left_image, size=(args.image_height, args.image_width), mode='bilinear', align_corners=False)
        right_image = torch.nn.functional.interpolate(right_image, size=(args.image_height, args.image_width), mode='bilinear', align_corners=False)

        # reverse_gamma + denoising
        night_images = torch.cat([left_image, right_image], dim = 0)
        denoised_night_images = denoise(night_images, diffusion_model, noise_scheduler)
        left_image, right_image = denoised_night_images[:1],denoised_night_images[1:]


        pair_image = torch.nn.functional.interpolate(pair_image, size=(args.image_height, args.image_width), mode='bilinear', align_corners=False)
        pair_stereo_pair = torch.nn.functional.interpolate(pair_stereo_pair, size=(args.image_height, args.image_width), mode='bilinear', align_corners=False)

        left_image = torch.cat([left_image, pair_image],dim = 0)
        right_image = torch.cat([right_image, pair_stereo_pair], dim = 0)


        #patched_left_image = patchify_images(left_image)
        #patched_right_image = patchify_images(right_image)

        with torch.no_grad():
            outputs = depth_net(left_image, right_image, return_dict=True)
            #outputs = depth_net(patched_left_image, patched_right_image, return_dict=True)
            predicted_disparities = outputs["flow_preds"]
            nn_distances = outputs["nn_distance"]
            masks = outputs["bad_pixel_mask"]
        
        disp = predicted_disparities[0][0:1].detach().cpu().numpy()
        # disp = predicted_disparities[0]
        # disp = disp.unsqueeze(1)
        # patch_shape = disp.shape[2:]
        # disp = disp.view(1, 2, 2, 1, patch_shape[0], patch_shape[1]).permute(0, 3, 1, 4, 2, 5).contiguous().view(1, 1, args.image_height, args.image_width)
        # disp = disp.squeeze(1).detach().cpu().numpy()
        predictions.append(disp)


        feat = outputs['features']
        idx = 0#torch.randint(0,384,(1,)).item()
        night_feat = feat[1][0][idx].detach().cpu().numpy()
        day_feat = feat[1][1][idx].detach().cpu().numpy()
        night_image = left_image[0:1].squeeze().permute(1,2,0).cpu().numpy()
        day_image = left_image[1:2].squeeze().permute(1,2,0).cpu().numpy()

        plot_features(night_image, night_feat, day_image, day_feat)

 

        breakpoint()


                
        # timestamp = data[reference_key]["timestamp"][0]
        # left_image = left_image[0:1].squeeze().permute(1,2,0).cpu().numpy()
        # left_image = put_text(left_image, str(timestamp))

        # fig, ax = plt.subplots(1,2, figsize=(10,5))
        # ax[0].imshow(left_image)
        # ax[1].imshow(disp.squeeze(), cmap='plasma')
        # #ax[2].imshow(masks[0].squeeze().detach().cpu().numpy(), cmap='plasma')
        # ax[0].axis('off')
        # ax[1].axis('off')
        # #ax[2].axis('off')
        # plt.tight_layout()
        # plt.savefig('test.png')    
        
        # breakpoint()

    #breakpoint()
    predictions = np.concatenate(predictions, axis=0)
    save_path = os.path.join(save_dir, 'baseline_d_n_f16_warping.npy')
    np.save(save_path, predictions)
    

if __name__ == "__main__":

    from datasets.robotcar.depth_test_dataloder import RobotcarTest
    #from datasets.ms2.depth_test_dataloader import MS2Dataset as RobotcarTest
    #from utils.options import get_test_args

    #args = get_test_args()
    import argparse
    parser = argparse.ArgumentParser(description='Feature-based SLAM')
    args = parser.parse_args()
    args.image_height = 192#384
    args.image_width = 320#640
    args.working_resolution = (args.image_width, args.image_height)
    args.use_gt_poses = False
    args.use_gray_scale = False
    args.use_stereo = True
    args.batch_size = 1
    args.use_pair = True
    args.split = "train"
    #args.test_file_path = ("/home/madhu/code/feature-slam/datasets/robotcar/2014-12-16-18-44-24_test.txt")
    args.test_file_path = '/home/madhu/code/feature-slam/git_repos/2014-12-16-18-44-24_paried_day_test.txt'
    args.data_path = '/mnt/nas/madhu/data/robotcar/2014-12-16-18-44-24/test_split/'
    args.pair_data_path = '/mnt/nas/madhu/data/robotcar/2014-12-09-13-21-02/test_split/'
    save_dir = '/mnt/nas/madhu/data/predictions/chapter_4_cvpr/'

    args.learning_rate = 1e-4

    dataset = RobotcarTest(args)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )
    
    depth_net = StereoDepthNet(args, reg_refine=False)
    depth_net.cuda()
    depth_net.model.eval()

    diffusion_model, noise_scheduler = get_model()
    print('Loaded the diffusion model and the noise schedular')

    #checkpoint_path = '/mnt/nas/madhu/data/checkpoints/chapter_4_cvpr/icra_2024_reproduce/depth_net_19.pth'
    #checkpoint_path = '/mnt/nas/madhu/data/checkpoints/chapter_4_cvpr/d_n_same_transfer_fusion_fp_16/depth_net_20.pth'
    checkpoint_path = '/mnt/nas/madhu/data/checkpoints/chapter_4_cvpr/d_n_same_transfer_fusion_fp_16_v2/depth_net_15.pth'
    checkpoint = torch.load(checkpoint_path)
    depth_net.load_state_dict(checkpoint,strict=True)


    test()