import os
import numpy as np
import torch
import copy
from PIL import Image
from datasets.base.camera import StereoCamera, MonoCamera
from torchvision.transforms.functional import rgb_to_grayscale
#from .transform import build_se3_transform
#from .interpolate_poses import interpolate_ins_poses
from datasets.base.augmentations import DataAugmentationDINO

original_resolution = [1280, 768]
ins_pose = -1.7132, 0.1181, 1.1948, -0.0125, 0.0400, 0.0050
#T_camera_posesource = build_se3_transform(ins_pose)
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


USE_DINO_AUGMENTATION = False
class RobotCarDataset(torch.utils.data.Dataset):
    def __init__(self, args):

        self.args = args
        camera0_info = get_camera_matrix(scale_x=1.0, scale_y=1.0)
        self.baseline_distance = 0.2
        # load timestamps
        self.timestamps = self.load_timestamps(self.args.data_path)

        # camera realted initailization
        if args.use_stereo:
            camera1_info = copy.deepcopy(camera0_info)
            camera1_info["name"] = "right_rgb"
            camera_info = {
                "camera_0": camera0_info,
                "camera_1": camera1_info,
                "baseline_distance": self.baseline_distance,
            }
            self.camera_rig = StereoCamera(camera_info)
        else:
            self.camera_rig = MonoCamera(camera0_info)

        self.camera_rig.set_data_path(args.data_path)
        self.camera_rig.set_working_resolution(args.working_resolution)

        if args.use_gt_poses:
            # vo_path = os.path.join(args.data_path, 'vo/vo.csv')
            #vo_path = os.path.join(args.data_path, "gps/ins.csv")
            print('Using GT Poses - RTK data!!!')
            vo_path = os.path.join(args.data_path, "gps/rtk.csv")
            self.load_ground_truth_poses(vo_path)
            # assert len(self.ground_truth_poses) == len(self.timestamps) + (self.seq_length -1)

        if USE_DINO_AUGMENTATION:
            print('Using DINO Augmentation!!!')
            self.dino_augmentation = DataAugmentationDINO((0.4,1), (0.05,4),8)

    def load_timestamps(self, data_path):

        if self.args.split == "train":
            print("Training!!!")
            # if self.args.use_stereo:
            #     timestamps_path = os.path.join(data_path, "files/train_stereo.txt")
            # else:
            timestamps_path = os.path.join(data_path, "../files/2014-12-16-18-44-24_train_1m.txt")
            #timestamps_path = os.path.join(data_path, "../files/2014-12-09-13-21-02_train_1m.txt")
        elif self.args.split == "val":
            print("Validation!!!")
            timestamps_path = os.path.join(data_path, "files/val.txt")
        elif self.args.split == "test":
            print("Testing!!!")
            timestamps_path = os.path.join(data_path, "files/test.txt")

        elif self.args.split == "pose_test":
            print("Pose Testing!!!")
            timestamps_path = os.path.join(
                data_path, "files/2014-12-16-18-44-24_pose_test.txt"
            )

        elif self.args.split == 'superpoint_fail':
            print('Using Superpoint Failures!!!')
            timestamps_path = os.path.join('/home/madhu/code/feature-slam/datasets/robotcar/superpoint_failure_triplets.txt')

        stamps = np.loadtxt(timestamps_path, dtype=int)
        assert len(stamps.shape) == 2
        stamps = stamps[: -(self.args.seq_length * self.args.stride)]

        return stamps

    def __len__(self):
        return len(self.timestamps)

    def get_multi_scale_images(self, image, num_scales=4):

        output_dict = []
        for i in range(num_scales):
            scale = 2**i
            cur_image = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                scale_factor=1 / scale,
                mode="bilinear",
                align_corners=True,
                recompute_scale_factor=True,
            ).squeeze(0)
            output_dict.append(cur_image)

        return output_dict

    def get_multi_scale_camera_matrix(self, camera_matrix, scales=4):

        output_dict = []
        for i in range(scales):
            scale = 2**i
            cur_camera_matrix = camera_matrix.clone()
            cur_camera_matrix[0, :] /= scale
            cur_camera_matrix[1, :] /= scale
            output_dict.append(cur_camera_matrix)

        return output_dict

    def load_ground_truth_poses(self, path):

        unique_timestamps = np.unique(self.timestamps)
        start_time = unique_timestamps[0]
        # poses = interpolate_vo_poses(path,unique_timestamps, start_time)
        if 'rtk' in path:
            poses = interpolate_ins_poses(
                path, unique_timestamps, start_time, use_rtk=True
            )
        else:
            poses = interpolate_ins_poses(
                path, unique_timestamps, start_time, use_rtk=False
            )
        poses = np.array(poses)
        # print('poses shape', poses.shape)
        pose_dict = {}
        for stamp, pose in zip(unique_timestamps, poses):
            pose_dict[str(stamp)] = pose
        self.ground_truth_poses = pose_dict

    def get_pose_at(self, key):

        #np.save('ground_truth_poses.npy', self.ground_truth_poses)

        # we need to convert these into kitti coordinate system for ease of use
        pose = self.ground_truth_poses[key]


        if pose.shape[0] == 4:
            pose = T_camera_posesource @ pose
            return torch.from_numpy(pose).float()
        else:
            # make it homogeneous
            pose = np.vstack((pose, np.array([0, 0, 0, 1])))
            pose = T_camera_posesource @ pose
            pose = torch.from_numpy(pose).float()
            return pose

    def load_data_at(self, timestamp):

        output_dict = {}

        full_resolution = self.args.use_full_res

        if self.args.use_stereo:

            image, stereo_pair = self.camera_rig.capture_image(
                timestamp,
                undistort=self.args.undistort,
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
            if self.args.use_gray_scale:
                gray_image = rgb_to_grayscale(image)
                gray_stereo_pair = rgb_to_grayscale(stereo_pair)
                output_dict["gray_image"] = gray_image
                output_dict["gray_stereo_pair"] = gray_stereo_pair

            stereo_pair = torch.from_numpy(stereo_pair).permute(2, 0, 1).float()
            camera_matrix0 = self.camera_rig.camera_0.camera_matrix
            camera_matrix1 = self.camera_rig.camera_1.camera_matrix
            camera_matrix0 = torch.from_numpy(camera_matrix0).float()
            camera_matrix1 = torch.from_numpy(camera_matrix1).float()

            # get multi scale images
            if self.args.use_multi_scale_images:
                multi_scale_images = self.get_multi_scale_images(
                    image, num_scales=self.args.num_scales
                )
                multi_scale_stereo_pair = self.get_multi_scale_images(
                    stereo_pair, num_scales=self.args.num_scales
                )
                multi_scale_camera_matrix0 = self.get_multi_scale_camera_matrix(
                    camera_matrix0, scales=self.args.num_scales
                )
                multi_scale_camera_matrix1 = self.get_multi_scale_camera_matrix(
                    camera_matrix1, scales=self.args.num_scales
                )

                output_dict["multi_scale_images"] = multi_scale_images
                output_dict["multi_scale_stereo_pair"] = multi_scale_stereo_pair
                output_dict["multi_scale_camera_matrix"] = multi_scale_camera_matrix0
                output_dict["multi_scale_camera_matrix1"] = multi_scale_camera_matrix1

            else:

                output_dict["image"] = image
                output_dict["stereo_pair"] = stereo_pair
                output_dict["camera_matrix"] = camera_matrix0
                output_dict["camera_matrix1"] = camera_matrix1

        else:
            image = self.camera_rig.capture_image(
                timestamp, undistort=self.args.undistort
            )


            if USE_DINO_AUGMENTATION:
                pil_image = Image.fromarray(np.uint8(image)).convert('RGB')
                dino_augmented_images = self.dino_augmentation(pil_image)
                output_dict["dino_augmented_images"] = dino_augmented_images

            image = torch.from_numpy(image).permute(2, 0, 1).float()
            if self.args.use_gray_scale:
                gray_image = rgb_to_grayscale(image)
                output_dict["gray_image"] = gray_image

            camera_matrix = torch.from_numpy(
                self.camera_rig.camera.camera_matrix
            ).float()

            # get multi scale images
            if self.args.use_multi_scale_images:
                multi_scale_images = self.get_multi_scale_images(
                    image, num_scales=self.args.num_scales
                )
                multi_scale_camera_matrix = self.get_multi_scale_camera_matrix(
                    camera_matrix, scales=self.args.num_scales
                )
                output_dict["multi_scale_images"] = multi_scale_images
                output_dict["multi_scale_camera_matrix"] = multi_scale_camera_matrix
            else:
                output_dict["image"] = image
                output_dict["camera_matrix"] = camera_matrix

        return output_dict

    def __getitem__(self, index):

        neighbouring_timestamps = self.timestamps[index]
        output_dict = {}

        if self.args.use_seq:

            # end =   index + (self.args.seq_length * self.args.stride)
            # neighbouring_timestamps = self.timestamps[index:end][::self.args.stride]
            # print(neighbouring_timestamps, self.args.seq_length)
            neighbouring_timestamps = [
                "{:010d}".format(int(i)) for i in neighbouring_timestamps
            ]

            for i, timestamp in enumerate(neighbouring_timestamps):
                frame_data = self.load_data_at(timestamp)
                frame_data["timestamp"] = timestamp
                if self.args.use_gt_poses:
                    frame_data["pose"] = self.get_pose_at(timestamp)

                output_dict["frame{}".format(i)] = frame_data

        else:
            frame_data = self.load_data_at("{:010d}".format(neighbouring_timestamps[1]))
            if self.args.use_gt_poses:
                frame_data["pose"] = self.get_pose_at(neighbouring_timestamps[1])
            output_dict["frame0"] = frame_data

        output_dict["dataset_index"] = index
        output_dict['timestamps'] = neighbouring_timestamps
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
    args.split = "train"

    args.data_path = "/hdd1/madhu/data/robotcar/2014-12-16-18-44-24/stereo/"
    dataset = RobotCarDataset(args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    all_xyz = []

    import tqdm
    for idx, data in tqdm.tqdm(enumerate(dataloader),total = len(dataloader)):

        pass

        
