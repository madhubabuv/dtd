import os
import numpy as np
import torch
import cv2
import tqdm



original_resolution = [1280, 768]

def get_camera_matrix(scale_x=0.25, scale_y=0.25):

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

    return torch.from_numpy(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])).float()

class DayNightDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        #timestamps_path = '/home/madhu/code/feature-slam/git_repos/dtd/notebooks/unique_paired_night_day.txt'
        timestamps_path = '/home/madhu/code/feature-slam/git_repos/2014-12-16-18-44-24_paried_day_train_1m.txt'
        self.day_data_path = os.path.join(self.data_path, '2014-12-09-13-21-02/stereo')
        self.night_data_path = os.path.join(self.data_path, '2014-12-16-18-44-24/stereo')
        self.timestamps = np.loadtxt(timestamps_path, dtype = str, delimiter = ' ')

    def __len__(self):
        return len(self.timestamps)

    def load_image(self, img_path):
        assert os.path.exists(img_path), "Image path {} does not exist".format(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.args.image_width, self.args.image_height))
        img = img / 255.0
        img = torch.from_numpy(img).float()
        img = img.permute(2, 0, 1)
        return img



    def __getitem__(self, idx):

        seq = self.timestamps[idx]
    
        night_stamp, day_stamp = seq[0], seq[1]
        night_left_img_path = os.path.join(self.night_data_path, 'left_rgb/data', night_stamp + '.png')
        day_left_img_path = os.path.join(self.day_data_path, 'left_rgb/data', day_stamp + '.png')

        night_right_img_path = os.path.join(self.night_data_path, 'right_rgb/data', night_stamp + '.png')
        day_right_img_path = os.path.join(self.day_data_path, 'right_rgb/data', day_stamp + '.png')

        assert os.path.exists(night_left_img_path), "Image path {} does not exist".format(night_left_img_path)
        assert os.path.exists(day_left_img_path), "Image path {} does not exist".format(day_left_img_path)
        assert os.path.exists(night_right_img_path), "Image path {} does not exist".format(night_right_img_path)
        assert os.path.exists(day_right_img_path), "Image path {} does not exist".format(day_right_img_path)

        night_left_img = self.load_image(night_left_img_path)
        day_left_img = self.load_image(day_left_img_path)
        
        night_right_img = self.load_image(night_right_img_path)
        day_right_img = self.load_image(day_right_img_path)
        

        outputs = {}
        frame0 = {}
        frame0['image'] = night_left_img
        frame0['stereo_pair'] = night_right_img
        frame0['camera_matrix'] = get_camera_matrix()

        frame1 = {}
        frame1['image'] = day_left_img
        frame1['stereo_pair'] = day_right_img
        frame1['camera_matrix'] = get_camera_matrix()

        outputs['frame0'] = frame0
        outputs['frame1'] = frame1
        return outputs


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Minimal RobotCar dataloader')
    args = parser.parse_args()

    args.split = 'train'
    args.data_path = "/hdd1/madhu/data/robotcar"

    dataset = DayNightDataset(args)
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
