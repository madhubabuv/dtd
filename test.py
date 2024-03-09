import tqdm
import numpy as np
import torch
import os
import cv2

import argparse
from models.depth_model import StereoDepthNet
from matplotlib import pyplot as plt
from datasets.robotcar.depth_test_dataloder import RobotcarTest
from datasets.ms2.depth_test_dataloader import MS2Dataset 

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
        with torch.no_grad():
            disp, distances, masks = depth_net(left_image, right_image,norm = False, masks = True,return_distance = True)        
        disp = disp[0].detach().cpu().numpy()
        predictions.append(disp)
        timestamp = data[reference_key]["timestamp"][0]
        left_image = left_image.squeeze().permute(1,2,0).cpu().numpy()
        left_image = put_text(left_image, str(timestamp))
        
        if args.debug:
            fig, ax = plt.subplots(1,3, figsize=(15,4))
            ax[0].imshow(left_image)
            ax[1].imshow(disp.squeeze(), cmap='plasma')
            ax[2].imshow(masks[0].squeeze().detach().cpu().numpy(), cmap='plasma')
            ax[0].axis('off')
            ax[1].axis('off')
            ax[2].axis('off')
            plt.tight_layout()
            plt.savefig('test.png')    
            breakpoint()

    predictions = np.concatenate(predictions, axis=0)
    save_path = os.path.join(args.save_dir, 'predictions_20.npy')
    np.save(save_path, predictions)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Feature-based SLAM')
    parser.add_argument('--data_path', type=str, help='Path to the test data')
    parser.add_argument('--test_file_path', type=str, help='Path to the test file',
                                           default ='datasets/robotcar/files/2014-12-16-18-44-24_test.txt')
    parser.add_argument('--image_height', type=int, default=192, help='Image height')
    parser.add_argument('--image_width', type=int, default=320, help='Image width')
    parser.add_argument('--dataset_name', type=str, default = 'robotcar', help='Dataset to use', choices = ['robotcar', 'ms2'])
    parser.add_argument('--debug', action='store_true', help='Debug mode, the programme stop after a single iteration generating an image with input image, disparity and mask')
    parser.add_argument('--checkpoint_path', type = str, help='checkpoint path', required=True)
    parser.add_argument('--save_dir', type=str,
                        help='Path to save the predictions', 
                        default = 'predictions/')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    #args.data_path = '/mnt/nas/madhu/data/robotcar/2014-12-16-18-44-24/test_split/'
    # args.test_file_path = '/hdd1/madhu/data/ms2/test_nighttime_list.txt'
    # args.data_path = "/hdd1/madhu/data/ms2"

    args.learning_rate = 1e-4
    dataset = RobotcarTest(args)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False
    )
    
    depth_net = StereoDepthNet(args)
    depth_net.cuda()
    depth_net.model.eval()

    #checkpoint_path = '/mnt/nas/madhu/data/checkpoints/chapter_3/MS2_dino_unimatch_v2_smooth_0.1/depth_net_10.pth'
    #checkpoint_path = '/mnt/nas/madhu/data/checkpoints/chapter_3/dino_unimatch_v2_smooth_0.1/depth_net_10.pth'
    checkpoint = torch.load(args.checkpoint_path)
    depth_net.load_state_dict(checkpoint, strict=False)

    #[0.18000043 2.04835197 7.21539604 0.27969543 0.7402708  0.89602731 0.94949014]
    test()