import torch
import os
import numpy as np
from datasets.base.image import Image
import cv2

class MS2Dataset(torch.utils.data.Dataset):
    def __init__(self, args):

        data_path = args.data_path
        test_file_path = args.test_file_path
        working_resolution = args.working_resolution
        use_stereo = args.use_stereo

        self.test_file_path = test_file_path
        self.test_files = self.read_test_files(test_file_path)

        self.timestamps = self.test_files[:, 0]

        self.use_stereo = use_stereo

        self.camera_intrinsics = self.load_camera_intrinsics(data_path +'/intrinsic_left.npy')

        self.working_resolution = working_resolution

    def read_test_files(self, test_file_path):

        data = np.loadtxt(test_file_path, dtype=str, delimiter=",")

        return data

    def load_camera_intrinsics(self, intrinsic_path):
            
        K = np.load(intrinsic_path)
        K[0,2] = 303.03 
        K[0,:] = K[0,:] * 0.5
        K[1,:] = K[1,:] * 0.5
        return torch.from_numpy(K).float()

    def __len__(self):

        return len(self.test_files)

    def read_image(self, image_path):


        image = Image.read(image_path)
        image = Image.to_rgb(image)
        image = Image.normalize(image)
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        height, width = image.shape[1], image.shape[2]
        new_height = 384
        new_width = 640
        #center crop
        left = (width - new_width)//2
        top = (height - new_height)//2
        right = (width + new_width)//2
        bottom = (height + new_height)//2

        image = image[:, top:bottom, left:right]

        #image = cv2.resize(image, self.working_resolution)

        return image

    def split_path_into_folder_and_file(self, path):
        split_path = path.split("/")
        folder = split_path[-4]
        file = split_path[-1].split(".")[0]
        
        return folder+'  '+file


    def __getitem__(self, idx):

        image_path = self.test_files[idx]
        
        timestamp = self.split_path_into_folder_and_file(image_path[0])

        image = self.read_image(image_path[0])
        if self.use_stereo:
            stereo_image = self.read_image(image_path[1])
            return {
                "frame0": {
                    "image": image,
                    "stereo_pair": stereo_image,
                    'timestamp': timestamp,
                    'camera_matrix':self.camera_intrinsics
                }
            }

        return {"frame0": {"image": image, "timestamp":timestamp}}


if __name__ == "__main__":

    from utils.options import get_test_args
    args = get_test_args()
    args.test_file_path = '/hdd1/madhu/data/ms2/test_nighttime_list.txt'
    args.use_stereo = True
    args.data_path = "/hdd1/madhu/data/ms2"

    dataset = MS2Dataset(args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for data in dataloader:

        img0 = data['frame0']['image']
        img1 = data['frame0']['stereo_pair']

        from matplotlib import pyplot as plt
        plt.imsave('img0.png', img0[0].permute(1,2,0).numpy())
        plt.imsave('img1.png', img1[0].permute(1,2,0).numpy())

        breakpoint()