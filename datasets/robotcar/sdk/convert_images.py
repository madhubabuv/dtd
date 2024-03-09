# The original code is taken from the robotcar-dataset-sdk and modified to work in parallael 
import sys
sys.path.append('robotcar-dataset-sdk/python')
import re
import tqdm
import os
import cv2
import numpy as np
import argparse
from camera_model import CameraModel
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic
from multiprocessing.pool import ThreadPool
from pathlib import Path
BAYER_STEREO = 'gbrg'
BAYER_MONO = 'rggb'

def load_image(image_path, model=None):
    """Loads and rectifies an image from file.

    Args:
        image_path (str): path to an image from the dataset.
        model (camera_model.CameraModel): if supplied, model will be used to undistort image.

    Returns:
        numpy.ndarray: demosaiced and optionally undistorted image

    """
    if model:
        camera = model.camera
    else:
        camera = re.search('(stereo|mono_(left|right|rear))', image_path).group(0)
    if camera == 'stereo':
        pattern = BAYER_STEREO
    else:
        pattern = BAYER_MONO

    img = Image.open(image_path)
    img = demosaic(img, pattern)
    if model:
        img = model.undistort(img)

    return img


def read_and_save(timestamp):
    save_image_path = os.path.join(save_image_dir, str(timestamp) + '.png')
    if os.path.exists(save_image_path):
         pass
    else:
        filename = os.path.join(args.data_path, str(timestamp) + '.png')
        img = load_image(filename, camera_model)
        if args.keep_car_hood:
            cropped_img = img[:768, :,:] # crop the car hood 1/5th of the image height 4*960 / 5 = 768
        else:
            cropped_img = img
        re_img = cv2.resize(cropped_img, (args.image_width, args.image_height))
        re_img = re_img.astype(np.uint8)
        re_img = cv2.cvtColor(re_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_image_path, re_img)

if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Convert images to RGB')
    parser.add_argument('--data_path', type=str, help='Path to the timestamps file',required=True)
    parser.add_argument('--models_dir', type=str, help='Path to the camera models directory',default=
                            'robotcar-dataset-sdk/models/')
    parser.add_argument('--timestamp_path', type=str, help='Path to the timestamps file', default=None)
    parser.add_argument('--save_image_dir', type=str, help='Path to save the RGB images', default = None)   
    parser.add_argument('--keep_car_hood',action='store_false', help='Crop the car hood from the images')
    parser.add_argument('--image_height', type=int, help='Height of the images', default=384)
    parser.add_argument('--image_width', type=int, help='Width of the images', default=640)
    args = parser.parse_args()

    # Arugments
    print('-------------------')
    print("Arguments:")
    print('-------------------')
    for arg in vars(args):
        print(arg,":",getattr(args, arg))
    print('-------------------')

    assert 'right' in args.data_path or 'left' in args.data_path, 'Please provide the path until stereo/right' 
    if args.timestamp_path is None:
        print('Using the files in the directory to get the timestamps')
        #files
        all_files = os.listdir(args.data_path)
        assert len(all_files)>0, 'No files found in the directory'
        data = [Path(f).stem for f in all_files if f.endswith('.png')]
        #base_dir = os.path.join(args.data_path,'../../')
        #timestamps_path = os.path.join(base_dir, 'stereo.timestamps')
    else:
        data = np.loadtxt(args.timestamps_path, dtype = int)
        if len(data.shape)>1:
            print('There are multiple columns, by default using the first one')
            data = data[:,0]

    if args.save_image_dir is None:
        stereo_base_dir = os.path.join(args.data_path,'../')
        if 'right' in args.data_path:
            save_image_dir = os.path.join(stereo_base_dir, 'right_rgb/data')
        else:
            save_image_dir = os.path.join(stereo_base_dir, 'left_rgb/data')

    if not os.path.exists(save_image_dir):
        print('Creating the models directory')
        os.makedirs(save_image_dir)

    camera_model = CameraModel(args.models_dir, args.data_path)

    if not os.path.exists(save_image_dir):
        os.makedirs(save_image_dir)

    try:
        with ThreadPool() as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(read_and_save, data), total=len(data)):
                pass

        print('All images in train set are converted to RGB and saved in {}'.format(save_image_dir))
    except Exception as e:
        print(e)
        breakpoint()
    

