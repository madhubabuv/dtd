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


def denoise_dataset():
    for idx, data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        reference_idx = 0
        reference_key = "frame{}".format(reference_idx)

        left_image = data[reference_key]["image"].cuda()
        right_image = data[reference_key]["stereo_pair"].cuda()


        #left_image = torch.nn.functional.interpolate(left_image, size=(args.image_height, args.image_width), mode='bilinear', align_corners=False)
        #right_image = torch.nn.functional.interpolate(right_image, size=(args.image_height, args.image_width), mode='bilinear', align_corners=False)

        # reverse_gamma + denoising
        night_images = torch.cat([left_image, right_image], dim = 0)
        
        denoised_night_images = denoise(night_images, diffusion_model, noise_scheduler)
        de_left_image, de_right_image = denoised_night_images[:args.batch_size],denoised_night_images[args.batch_size:]


        for ind in range(args.batch_size):
            timestamp = data[reference_key]["timestamp"][ind]
            left_image = de_left_image[ind:ind+1].squeeze().permute(1,2,0).cpu().numpy()
            right_image = de_right_image[ind:ind+1].squeeze().permute(1,2,0).cpu().numpy()
            
            left_image_path = os.path.join(left_image_dir,str(timestamp)+'.png')
            right_image_path = os.path.join(right_image_dir,str(timestamp)+'.png')

            plt.imsave(left_image_path, left_image)
            plt.imsave(right_image_path, right_image)
        
        #breakpoint()


if __name__ == "__main__":

    from datasets.robotcar.day_night_paried_dataset import DayNightDataset as RobotCarDataset
    data_path = '/hdd1/madhu/data/robotcar/2014-12-16-18-44-24_ddpm_192x320/stereo/'
    left_image_dir = os.path.join(data_path, 'left_rgb/data')
    right_image_dir = os.path.join(data_path,'right_rgb/data')
    if not os.path.exists(left_image_dir):
        os.makedirs(left_image_dir)
    if not os.path.exists(right_image_dir):
        os.makedirs(right_image_dir)

    import argparse
    parser = argparse.ArgumentParser(description='Feature-based SLAM')
    args = parser.parse_args()
    args.image_height = 192#384
    args.image_width = 320#640
    args.working_resolution = (args.image_width, args.image_height)
    args.use_gt_poses = False
    args.use_gray_scale = False
    args.use_stereo = True
    args.batch_size = 24
    args.use_pair = True
    args.split = "train"
    args.data_path = "/hdd1/madhu/data/robotcar"
    dataset = RobotCarDataset(args)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,num_workers = 32
    )


    diffusion_model, noise_scheduler = get_model()
    print('Loaded the diffusion model and the noise schedular')

    denoise_dataset()