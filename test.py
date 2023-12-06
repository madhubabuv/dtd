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

def test():
    depth_net.eval()
    depth_net.model.eval()
    predictions = []
    for idx, data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        reference_idx = 0
        reference_key = "frame{}".format(reference_idx)
        left_image = data[reference_key]["image"].cuda()
        right_image = data[reference_key]["stereo_pair"].cuda()
        left_image = torch.nn.functional.interpolate(left_image, size=(args.image_height, args.image_width), mode='bilinear', align_corners=False)
        right_image = torch.nn.functional.interpolate(right_image, size=(args.image_height, args.image_width), mode='bilinear', align_corners=False)

        #patched_left_image = patchify_images(left_image)
        #patched_right_image = patchify_images(right_image)

        with torch.no_grad():
            outputs = depth_net(left_image, right_image, return_dict=True)
            #outputs = depth_net(patched_left_image, patched_right_image, return_dict=True)
            predicted_disparities = outputs["flow_preds"]
            nn_distances = outputs["nn_distance"]
            masks = outputs["bad_pixel_mask"]
        
        disp = predicted_disparities[0].detach().cpu().numpy()
        # disp = predicted_disparities[0]
        # disp = disp.unsqueeze(1)
        # patch_shape = disp.shape[2:]
        # disp = disp.view(1, 2, 2, 1, patch_shape[0], patch_shape[1]).permute(0, 3, 1, 4, 2, 5).contiguous().view(1, 1, args.image_height, args.image_width)
        # disp = disp.squeeze(1).detach().cpu().numpy()
        predictions.append(disp)
                
        # timestamp = data[reference_key]["timestamp"][0]
        # left_image = left_image.squeeze().permute(1,2,0).cpu().numpy()
        # left_image = put_text(left_image, str(timestamp))

        # fig, ax = plt.subplots(1,2, figsize=(15,4))
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
    save_path = os.path.join(save_dir, 'test.npy')
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
    args.split = "train"
    args.test_file_path = ("/home/madhu/code/feature-slam/datasets/robotcar/2014-12-16-18-44-24_test.txt")
    args.data_path = '/mnt/nas/madhu/data/robotcar/2014-12-16-18-44-24/test_split/'
    save_dir = '/mnt/nas/madhu/data/predictions/chapter_4_cvpr/'

    args.learning_rate = 1e-4

    dataset = RobotcarTest(args)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )
    
    depth_net = StereoDepthNet(args, reg_refine=False)
    depth_net.cuda()
    depth_net.model.eval()

    #checkpoint_path = '/mnt/nas/madhu/data/checkpoints/chapter_4_cvpr/icra_2024_reproduce/depth_net_19.pth'
    checkpoint_path = '/mnt/nas/madhu/data/checkpoints/chapter_4_cvpr/d_n_v2_fp_16/depth_net_10.pth'
    checkpoint = torch.load(checkpoint_path)
    depth_net.load_state_dict(checkpoint,strict=False)


    test()