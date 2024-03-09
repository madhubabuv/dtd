import torch
import os
import cv2
import numpy as np
class RobotcarTest(torch.utils.data.Dataset):
    def __init__(self, args):
        print('RobotCar TEST DATASET')
        self.args = args
        self.test_files = self.read_test_files(self.args.test_file_path)
        self.timestamps = self.test_files[:, 0]
        self.timestamps_pair = self.test_files[:,1]

        assert os.path.exists(self.args.data_path), "Data path does not exist: {}".format(self.args.data_path)
        self.test_file_full_path = [os.path.join(self.args.data_path, "left", test_file[0] + ".png")for test_file in self.test_files]

        # assertion
        for test_file in self.test_file_full_path:
            assert os.path.exists(test_file), "Image does not exist: {}".format(test_file)
        self.args = args

    def read_test_files(self, test_file_path):
        data = np.loadtxt(test_file_path, dtype=str, delimiter=" ")
        return data

    def __len__(self):
        return len(self.test_files)
    def read_image(self, image_path):
        assert os.path.exists(image_path), "Image does not exist: {}".format(image_path)
        image = cv2.imread(image_path)

        shape = image.shape
        height, width = shape[0], shape[1]
        crop_height = (4 * height) // 5
        image = image[:crop_height, :, :]
        image = cv2.resize(image, (self.args.image_width, self.args.image_height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        return image

    def __getitem__(self, idx):

        image_path = self.test_file_full_path[idx]
        image = self.read_image(image_path)
        timestamp = self.timestamps[idx]
        frame0 = {"image": image, "timestamp": timestamp}
        stereo_image_path = image_path.replace("left", "right")
        stereo_image = self.read_image(stereo_image_path)
        frame0['stereo_pair'] = stereo_image
        return {'frame0':frame0}


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='DTD Depth Estimation')
    args = parser.parse_args()
    args.image_width = 320
    args.image_height = 192

    args.test_file_path = '/home/madhu/code/feature-slam/datasets/robotcar/2014-12-16-18-44-24_test.txt'
    args.data_path = "/mnt/nas/madhu/data/robotcar/2014-12-16-18-44-24/test_split/"
    dataset = RobotcarTest(args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for data in dataloader:
        img0 = data['frame0']['image']
        img1 = data['frame0']['stereo_pair']

        print(img0.shape)
        print(img1.shape)

        from matplotlib import pyplot as plt

        plt.imsave('img0.png', img0[0].permute(1,2,0).numpy())
        plt.imsave('img1.png', img1[0].permute(1,2,0).numpy())

        breakpoint()