import numpy as np
import cv2
import argparse
import os
import tqdm
import torch
from datasets.ms2.depth_test_dataloader import MS2Dataset as RobotcarTest
from matplotlib import pyplot as plt

def center_crop(image):

    height, width = image.shape[0], image.shape[1]
    new_height = 384
    new_width = 640
    #center crop
    left = (width - new_width)//2
    top = (height - new_height)//2
    right = (width + new_width)//2
    bottom = (height + new_height)//2

    image = image[top:bottom, left:right]

    return image

def test():


    all_depths = []


    for idx, data in tqdm.tqdm(enumerate(dataloader), total = len(dataloader)):

        key = 'frame0'
        #image  = data[key]['image']
        path_split = data[key]['timestamp'][0].split(' ')
        folder = path_split[0]
        img_id = path_split[-1]
        depth_path = os.path.join(args.data_path, 'proj_depth', folder, 'rgb/depth_filtered', img_id+'.png')
        assert os.path.exists(depth_path), depth_path
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        depth = depth.astype(np.float32)/256.0
        depth = center_crop(depth)

        all_depths.append(depth)

    


        # fig, ax = plt.subplots(1,2, figsize=(10,5))
        # ax[0].imshow(image[0].permute(1,2,0).numpy())
        # ax[1].imshow(depth, cmap='plasma')
        # ax[0].axis('off')
        # ax[1].axis('off')
        # plt.tight_layout()
        # plt.savefig('depth.png')

        # breakpoint()

    breakpoint()
    save_path = os.path.join(args.data_path, 'gt_test_depths_filtered.npy')
    np.save(save_path, all_depths)
    



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Feature-based SLAM')
    args = parser.parse_args()
    args.image_height = 384
    args.image_width = 640
    args.batch_size =1
    args.working_resolution = (args.image_width, args.image_height)
    args.use_stereo = True
    args.test_file_path = '/hdd1/madhu/data/ms2/test_nighttime_list.txt'
    args.data_path = "/hdd1/madhu/data/ms2"
    dataset = RobotcarTest(args)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )

    depth_path = os.path.join(args.data_path, 'proj_depth')


    test()
