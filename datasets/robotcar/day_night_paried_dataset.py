import os
import numpy as np
import torch
import cv2
import tqdm
from .interpolate_poses import interpolate_vo_poses, interpolate_ins_poses
from core.sfm.gauss_newton_nn_module import SE3_inverse
from torchvision.transforms.functional import rgb_to_grayscale


original_resolution = [1280, 768]

def get_camera_matrix(scale_x=0.5, scale_y=0.5):

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

    return torch.from_numpy(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))



class MinimalRobotCar(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.split = args.split
        timestamps_path = '/home/madhu/code/feature-slam/datasets/robotcar/day_night_image_pairs.txt'
        self.timestamps = np.loadtxt(timestamps_path, dtype=int, usecols=(0,2))
        day_vo_path = os.path.join(args.data_path, "day/2014-12-09-13-21-02/gps/rtk.csv")
        night_vo_path = os.path.join(args.data_path, "night/2014-12-16-18-44-24/gps/rtk.csv")
        self.day_ground_truth_poses = self.load_ground_truth_poses(day_vo_path)
        self.night_ground_truth_poses = self.load_ground_truth_poses(night_vo_path)

    def __len__(self):
        return len(self.timestamps)

    def load_ground_truth_poses(self, path):

        unique_timestamps = np.unique(self.timestamps)
        
        start_time = unique_timestamps[0]
        if 'rtk' in path:
            poses = interpolate_ins_poses(
                path, unique_timestamps, start_time, use_rtk=True
            )
        else:
            poses = interpolate_ins_poses(
                path, unique_timestamps, start_time, use_rtk=False
            )
        poses = np.array(poses)
        pose_dict = {}
        for stamp, pose in zip(unique_timestamps, poses):
            pose_dict[str(stamp)] = pose

        return pose_dict
        self.ground_truth_poses = pose_dict

    def get_pose_at(self, key, time = 'day'):

        #np.save('ground_truth_poses.npy', self.ground_truth_poses)

        # we need to convert these into kitti coordinate system for ease of use
        #pose = self.ground_truth_poses[key]
        if time == 'day':
            pose = self.day_ground_truth_poses[key]
        else:
            pose = self.night_ground_truth_poses[key]
        if pose.shape[0] == 4:
            return torch.from_numpy(pose).float()
        else:
            # make it homogeneous
            pose = np.vstack((pose, np.array([0, 0, 0, 1])))
            pose = torch.from_numpy(pose).float()
            return pose

    def load_image(self, timestamp, time = 'day'):

        if time == 'night':
            img_path = os.path.join(self.data_path, '{}/2014-12-16-18-44-24/rgb/data/{}.png'.format(time,timestamp))
        else:
            img_path = os.path.join(self.data_path, '{}/2014-12-09-13-21-02/rgb/{}.png'.format(time,timestamp))

        assert os.path.exists(img_path), "Image path {} does not exist".format(img_path)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = torch.from_numpy(img).float()
        img = img.permute(2, 0, 1)
        return img


    def __getitem__(self, idx):

        seq = self.timestamps[idx]
        frame0 = {}
        frame0['image'] = self.load_image(seq[0], time = 'night')
        frame0['gray_image'] = rgb_to_grayscale(frame0['image'])
        frame0['pose'] = self.get_pose_at(str(seq[0]), time = 'night')
        frame0["camera_matrix"] = get_camera_matrix()


        frame1 = {}
        frame1['image'] = self.load_image(seq[1], time = 'day')
        frame1['gray_image'] = rgb_to_grayscale(frame1['image'])
        frame1['pose'] = self.get_pose_at(str(seq[1]), time = 'day')
        
        outputs = {}
        outputs['frame0'] = frame0
        outputs['frame1'] = frame1
        return outputs


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Minimal RobotCar dataloader')
    args = parser.parse_args()

    args.split = 'train'
    args.data_path = "/mnt/nas/madhu/robotcar/"

    dataset = MinimalRobotCar(args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    all_xyz = []


    for idx, data in tqdm.tqdm(enumerate(dataloader), total = len(dataloader)):

        img0 = data["frame0"]["image"]
        img1 = data["frame1"]["image"]

        print(img0.shape)


        from matplotlib import pyplot as plt

        plt.imsave("img0.png", img0[0].permute(1, 2, 0).numpy())
        plt.imsave("img1.png", img1[0].permute(1, 2, 0).numpy())

        break

        # print(img0.shape, img1.shape)

        # pose0 = data['frame0']['pose']
        # pose1 = data['frame1']['pose']

        # xyz = pose0[:3,3]
        # all_xyz.append(xyz)

        # relative_pose = SE3_inverse(pose1) @ pose0

        # print(relative_pose)

        #break
