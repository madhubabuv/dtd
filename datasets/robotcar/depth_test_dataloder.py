import torch
import os
import numpy as np
from datasets.base.image import Image


class RobotcarTest(torch.utils.data.Dataset):
    def __init__(self, args):

        data_path = args.data_path
        test_file_path = args.test_file_path
        working_resolution = args.working_resolution
        use_stereo = args.use_stereo

        self.test_file_path = test_file_path
        self.test_files = self.read_test_files(test_file_path)

        self.timestamps = self.test_files[:, 0]

        self.use_stereo = use_stereo
        assert os.path.exists(data_path), "Data path does not exist: {}".format(
            data_path
        )
        self.test_file_full_path = [
            os.path.join(data_path, "left", test_file[0] + ".png")
            for test_file in self.test_files
        ]

        # assertion
        for test_file in self.test_file_full_path:
            assert os.path.exists(test_file), "Image does not exist: {}".format(
                test_file
            )

        self.working_resolution = working_resolution

    def read_test_files(self, test_file_path):

        data = np.loadtxt(test_file_path, dtype=str, delimiter=" ")

        return data

    def __len__(self):

        return len(self.test_files)

    def read_image(self, image_path):

        image = Image.read(image_path)

        # I need to remove the car hood, Ain't I?
        shape = image.shape
        height, width = shape[0], shape[1]
        crop_height = (4 * height) // 5
        image = image[:crop_height, :, :]

        # image = Image.resize(
        #     image, self.working_resolution[0], self.working_resolution[1]
        # )
        # cv2 reads images in BGR format, so we convert it to RGB
        image = Image.to_rgb(image)
        image = Image.normalize(image)
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        return image

    def __getitem__(self, idx):

        image_path = self.test_file_full_path[idx]
        image = self.read_image(image_path)
        timestamp = self.timestamps[idx]
        if self.use_stereo:
            stereo_image_path = image_path.replace("left", "right")
            stereo_image = self.read_image(stereo_image_path)
            return {
                "frame0": {
                    "image": image,
                    "stereo_pair": stereo_image,
                    "timestamp": timestamp,
                }
            }

        return {"frame0": {"image": image, "timestamp": timestamp}}


if __name__ == "__main__":

    #from utils.options import get_test_args
    args = get_test_args()
    args.test_file_path = '/home/madhu/code/feature-slam/datasets/robotcar/2014-12-16-18-44-24_test.txt'

    args.data_path = "/mnt/nas/madhu/data/robotcar/2014-12-16-18-44-24/test_split/"

    dataset = RobotcarTest(args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for data in dataloader:

        img0 = data['frame0']['image']
        img1 = data['frame0']['stereo_pair']

        from matplotlib import pyplot as plt

        plt.imsave('img0.png', img0[0].permute(1,2,0).numpy())
        plt.imsave('img1.png', img1[0].permute(1,2,0).numpy())

        break