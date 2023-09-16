import tqdm
import numpy as np
import torch
import os
import cv2
from models.depth_model import StereoDepthNet
from matplotlib import pyplot as plt


def put_text(image,text):

    np_image = image * 255
    np_image = np_image.astype(np.uint8)
    np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    cv2.putText(np_image,text,(180,20), cv2.FONT_HERSHEY_DUPLEX, 0.4,(255,255,255),1, cv2.LINE_AA)

    np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    np_image = np_image.astype(np.float32) / 255
    return np_image


def test():
    depth_net.eval()
    predictions = []
    for idx, data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        reference_idx = 0
        reference_key = "frame{}".format(reference_idx)
        left_image = data[reference_key]["image"].cuda()
        right_image = data[reference_key]["stereo_pair"].cuda()

        left_image = torch.nn.functional.interpolate(left_image, size=(192, 320), mode='bilinear', align_corners=False)
        right_image = torch.nn.functional.interpolate(right_image, size=(192, 320), mode='bilinear', align_corners=False)
        with torch.no_grad():
            disp, distances, masks = depth_net(left_image, right_image,norm = False, masks = True,return_distance = True)
            #disp = depth_net(left_image, right_image)
        
        disp = disp[0].detach().cpu().numpy()
        predictions.append(disp)
            
        # timestamp = data[reference_key]["timestamp"][0]
        # left_image = left_image.squeeze().permute(1,2,0).cpu().numpy()
        # left_image = put_text(left_image, str(timestamp))

        # fig, ax = plt.subplots(1,3, figsize=(15,4))
        # ax[0].imshow(left_image)
        # ax[1].imshow(disp.squeeze(), cmap='plasma')
        # ax[2].imshow(masks[0].squeeze().detach().cpu().numpy(), cmap='plasma')
        # ax[0].axis('off')
        # ax[1].axis('off')
        # ax[2].axis('off')
        # plt.tight_layout()
        # plt.savefig('test.png')    
        
        # breakpoint()

    #breakpoint()
    predictions = np.concatenate(predictions, axis=0)
    save_path = os.path.join(save_dir, 'MS2_dino_unimatch_v2_smooth_0.1_trained.npy')
    np.save(save_path, predictions)
    

if __name__ == "__main__":

    #from datasets.robotcar.depth_test_dataloder import RobotcarTest
    from datasets.ms2.depth_test_dataloader import MS2Dataset as RobotcarTest
    #from utils.options import get_test_args

    #args = get_test_args()
    import argparse
    parser = argparse.ArgumentParser(description='Feature-based SLAM')
    args = parser.parse_args()
    args.image_height = 384
    args.image_width = 640
    args.working_resolution = (args.image_width, args.image_height)
    args.use_gt_poses = False
    args.use_gray_scale = False
    args.use_stereo = True
    args.batch_size = 1
    args.split = "train"
    # args.test_file_path = ("/home/madhu/code/feature-slam/datasets/robotcar/2014-12-16-18-44-24_test.txt")
    # args.data_path = '/mnt/nas/madhu/data/robotcar/2014-12-16-18-44-24/test_split/'
    #args.test_file_path = '/mnt/nas/madhu/awsgpu2/datasets/robotcar/2014-12-09-13-21-02/2014-12-09-13-21-02_test.txt'
    #args.data_path = '/hdd1/madhu/data/robotcar/2014-12-09-13-21-02/stereo'

    args.test_file_path = '/hdd1/madhu/data/ms2/test_nighttime_list.txt'
    args.data_path = "/hdd1/madhu/data/ms2"

    save_dir = '/mnt/nas/madhu/data/predictions/baslines/'

    args.learning_rate = 1e-4

    dataset = RobotcarTest(args)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )
    
    depth_net = StereoDepthNet(args, dino = True, resume = False, reg_refine=False)
    #depth_net = StereoDepthNetDinoV2(args, dino = True, resume = False)

    #depth_net = StereoDepthNet(args)
    depth_net.cuda()
    depth_net.model.eval()

    checkpoint_path = '/mnt/nas/madhu/data/checkpoints/chapter_3/MS2_dino_unimatch_v2_smooth_0.1/depth_net_10.pth'
    #checkpoint_path = '/mnt/nas/madhu/data/checkpoints/chapter_3_final/dino_unimatch_v2_orignal_bf/depth_net_20.pth'
    checkpoint = torch.load(checkpoint_path)
    depth_net.load_state_dict(checkpoint,strict=True)


    test()