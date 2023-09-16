import os
import numpy as np
import torch
import cv2
import tqdm
from PIL import Image
from torchvision.transforms.functional import rgb_to_grayscale
from .interpolate_poses import interpolate_vo_poses, interpolate_ins_poses
from datasets.robotcar.transform import build_se3_transform
from .metric_seperation import getOxSplits
from torchvision import transforms as T

original_resolution = [1280, 768]
normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
to_tensor = T.ToTensor()
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


ins_pose = -1.7132, 0.1181, 1.1948, -0.0125, 0.0400, 0.0050
T_camera_posesource = build_se3_transform(ins_pose)

class MinimalRobotCar(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.split = args.split
        if self.split =='train':
            timestamps_path = os.path.join(self.data_path, 'files/{}.txt'.format(args.split))
        elif self.split =='pose_test':
            timestamps_path = os.path.join(self.data_path, 'files/{}.txt'.format('2014-12-16-18-44-24_pose_test'))
        #self.timestamps = np.loadtxt(timestamps_path, dtype=int)

        stem_path = '/mnt/nas/madhu/robotcar/night'
        traverse_id = '2014-12-16-18-44-24'
        split = 'test'
        sample_dist = args.seperation
        print('Using seperation of {} meters'.format(sample_dist))
        draw_polt = False
        cam = 'stereo'

        #timestamps_path = os.path.join(self.data_path, 'files/{}.txt'.format('pose_test_{}'.format(sample_dist)))
        split_outputs = getOxSplits(stem_path, split, traverse_id, draw_polt, cam = cam, sampleDist = sample_dist,verbose=True)
        timestamps = split_outputs[-1].astype(int)
        #print('Number of frames in the split: {}'.format(len(self.timestamps)))
        #to triplets
        self.timestamps = np.array([timestamps[i:i+3] for i in range(len(timestamps)-2)])
        vo_path = os.path.join(args.data_path, "gps/rtk.csv")
        self.load_ground_truth_poses(vo_path)
        #remove static frames
        self.remove_static_frames()
        #sample metric seperation
        self.sample_with_metric_seperation()

        print('Number of frames in the split: {}'.format(len(self.timestamps)))

        
    def sample_with_metric_seperation(self):
        original_timestamps = self.ground_truth_poses.keys()
        int_timestamps = np.array(list(original_timestamps)).astype(int)
        int_timestamps.sort()
        distances = []
        for i in range(len(int_timestamps)-1):
            pose0 = self.ground_truth_poses[str(int_timestamps[i])]
            pose1 = self.ground_truth_poses[str(int_timestamps[i+1])]
            relative_pose = np.linalg.pinv(pose1) @ pose0
            translation = relative_pose[:3, 3]
            dist = np.linalg.norm(translation)
            distances.append(dist)
        distances = np.array(distances)
        distances = np.insert(distances, 0, 0)
        # we need seperate them based on the seperation

        timestamps = []
        final_distances = []
        for i in tqdm.tqdm(range(len(int_timestamps)),total=len(int_timestamps), desc='Sampling with metric seperation'):
            origin = int_timestamps[i]
            for j in range(i+1, len(int_timestamps)):
                cum_dist = np.sum(distances[i:j])
                if cum_dist > self.args.seperation and cum_dist < (self.args.seperation+0.5):
                    destination = int_timestamps[j]
                    final_distances.append(cum_dist)
                    timestamps.append([origin, destination])
                    break        

        assert (sum(np.array(final_distances) > self.args.seperation + 0.5)) == 0
        self.timestamps = np.array(timestamps)

    def remove_static_frames(self):
        original_timestamps = self.ground_truth_poses.keys()
        new_pose_dict = {}
        int_timestamps = np.array(list(original_timestamps)).astype(int)
        int_timestamps.sort()
        distances = []
        for i in tqdm.tqdm(range(len(int_timestamps)-1),total=len(int_timestamps)-1, desc='Removing static frames'):
            pose0 = self.ground_truth_poses[str(int_timestamps[i])]
            pose1 = self.ground_truth_poses[str(int_timestamps[i+1])]
            relative_pose = np.linalg.pinv(pose1) @ pose0
            translation = relative_pose[:3, 3]
            dist = np.linalg.norm(translation)
            distances.append(dist)
        distances = np.array(distances)

        #remove static frames and making sure there are no jumps
        # we want to stick to the seperation distance, anythign that is static or a lot grater than the seperation distance is removed
        non_static_frames = np.where(np.logical_and(distances > 0.05, distances < (self.args.seperation + 0.5)))
        print('Number of static frames: {}'.format(len(distances) - len(non_static_frames)))
        non_static_timestamps = int_timestamps[non_static_frames]
        for stamp in non_static_timestamps:
            new_pose_dict[str(stamp)] = self.ground_truth_poses[str(stamp)]
        self.ground_truth_poses = new_pose_dict

       
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

    def load_image(self, timestamp):

        img_path = os.path.join(self.data_path, 'rgb/data/{}.png'.format(timestamp))
        assert os.path.exists(img_path), "Image path {} does not exist".format(img_path)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = torch.from_numpy(img).float()
        img = img.permute(2, 0, 1)
        return img

    def pil_load_image(self,timestamp):
        img_path = os.path.join(self.data_path, 'rgb/data/{}.png'.format(timestamp))
        assert os.path.exists(img_path), "Image path {} does not exist".format(img_path)

        img = Image.open(img_path)
        img = to_tensor(img)
        return img

        


    def __getitem__(self, idx):

        #idx = 562
        
        seq = self.timestamps[idx]

        frame0 = {}
        frame0['image'] = self.pil_load_image(seq[0])
        frame0['gray_image'] = rgb_to_grayscale(frame0['image'])
        frame0['pose'] = self.get_pose_at(str(seq[0])).float()
        frame0["camera_matrix"] = get_camera_matrix().float()
        frame0['timestamp'] = seq[0]

 
        frame1 = {}
        frame1['image'] = self.pil_load_image(seq[1])
        frame1['gray_image'] = rgb_to_grayscale(frame1['image'])
        frame1['pose'] = self.get_pose_at(str(seq[1])).float()
        frame1["camera_matrix"] = get_camera_matrix().float()
        frame1['timestamp'] = seq[1]

    

        # frame2 = {}
        # frame2['image'] = self.load_image(seq[2])
        # frame2['gray_image'] = rgb_to_grayscale(frame2['image'])
        # frame2['pose'] = self.get_pose_at(str(seq[2])).float()
        
        outputs = {}
        outputs['frame0'] = frame0
        outputs['frame1'] = frame1
        #outputs['frame2'] = frame2

        return outputs


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Minimal RobotCar dataloader')
    args = parser.parse_args()

    args.split = 'train'
    args.seperation = 0.5
    args.data_path = "/mnt/nas/madhu/robotcar/night/2014-12-16-18-44-24/"

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
