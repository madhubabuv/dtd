import os
import numpy as np
import torch
import copy
from PIL import Image
from datasets.base.camera import StereoCamera, MonoCamera
from torchvision.transforms.functional import rgb_to_grayscale

original_resolution = [1280, 768]

def get_camera_matrix(scale_x=1.0, scale_y=1.0):

    # scale is set to 2 because oxfor images are captured
    # in raw format, so interms of space and speed, I never converted them to the
    # original resolution, also, removed the car hood as it does not carry any useful info
    # that is why the principal point is not at the center of the image

    fx, fy, cx, cy = 983.044006, 983.044006, 643.646973, 493.378998
    cy *= 4 / 5
    fx = fx * scale_x
    fy = fy * scale_y
    cx = cx * scale_x
    cy = cy * scale_y

    camera_info = {}
    camera_info["focal_x"] = fx
    camera_info["focal_y"] = fy
    camera_info["principal_x"] = cx
    camera_info["principal_y"] = cy
    camera_info["resolution"] = original_resolution  # [1280, 768]
    camera_info["distortion_coefficients"] = None

    camera_info[
        "name"
    ] = "left_rgb"  # this is the last folder name where the images are saved

    return camera_info


class RobotCarDataset(torch.utils.data.Dataset):
    def __init__(self, args):

        self.args = args
        camera0_info = get_camera_matrix(scale_x=1.0, scale_y=1.0)
        self.baseline_distance = 0.239
        # load timestamps
        self.timestamps = self.load_timestamps()

        # camera realted initailization
        camera1_info = copy.deepcopy(camera0_info)
        camera1_info["name"] = "right_rgb"
        camera_info = {
            "camera_0": camera0_info,
            "camera_1": camera1_info,
            "baseline_distance": self.baseline_distance,
        }
        self.camera_rig = StereoCamera(camera_info)
       

        self.camera_rig.set_data_path(args.data_path)
        self.camera_rig.set_working_resolution(args.working_resolution)


    def load_timestamps(self):

        timestamps_path = ("datasets/robotcar/files/2014-12-16-18-44-24_train.txt")
        stamps = np.loadtxt(timestamps_path, dtype=int)
        stamps = stamps[:,1] # middle stamps        
        return stamps

    def __len__(self):
        return len(self.timestamps)


    def load_data_at(self, timestamp):

        output_dict = {}

        full_resolution = self.args.use_full_res

        #if self.args.use_stereo:
        image, stereo_pair = self.camera_rig.capture_image(
            timestamp,
            undistort=False,
            full_resolution=full_resolution,
        )

        if full_resolution:

            image, full_res_image = image
            stereo_pair, full_res_stereo_pair = stereo_pair

            output_dict["full_res_image"] = (
                torch.from_numpy(full_res_image).permute(2, 0, 1).float()
            )
            output_dict["full_res_stereo_pair"] = (
                torch.from_numpy(full_res_stereo_pair).permute(2, 0, 1).float()
            )
            output_dict["full_res_camera_matrix"] = torch.from_numpy(
                self.camera_rig.camera_0.full_res_camera_matrix
            ).float()

        image = torch.from_numpy(image).permute(2, 0, 1).float()

        stereo_pair = torch.from_numpy(stereo_pair).permute(2, 0, 1).float()
        camera_matrix0 = self.camera_rig.camera_0.camera_matrix
        camera_matrix1 = self.camera_rig.camera_1.camera_matrix
        camera_matrix0 = torch.from_numpy(camera_matrix0).float()
        camera_matrix1 = torch.from_numpy(camera_matrix1).float()

        output_dict["image"] = image
        output_dict["stereo_pair"] = stereo_pair
        output_dict["camera_matrix"] = camera_matrix0
        output_dict["camera_matrix1"] = camera_matrix1
        return output_dict

    def __getitem__(self, index):

        stamp = self.timestamps[index]
        output_dict = {}
        frame_data = self.load_data_at("{:010d}".format(stamp))
        output_dict["frame0"] = frame_data
        output_dict["dataset_index"] = index
        output_dict['timestamps'] = stamp
        return output_dict


if __name__ == "__main__":

    import argparse
    import copy

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.working_resolution = (640, 384)
    args.use_stereo = True
    args.undistort = False
    args.use_seq = False
    args.seq_length = 3
    args.stride = 1
    args.use_multi_scale_images = False
    args.num_scales = 4
    args.use_full_res = False
    args.use_gray_scale = False
    args.use_gt_poses = False
    args.data_path = "/hdd1/madhu/data/robotcar/2014-12-16-18-44-24/stereo/"
    dataset = RobotCarDataset(args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    all_xyz = []

    import tqdm
    for idx, data in tqdm.tqdm(enumerate(dataloader),total = len(dataloader)):

        pass

        
