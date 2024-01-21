################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Geoff Pascoe (gmp@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
###############################################################################

import re
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic
from camera_model import CameraModel

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
        filename = os.path.join(raw_image_dir, str(timestamp) + '.png')
        img = load_image(filename, camera_model)
        cropped_img = img[:768, :,:]
        re_img = cv2.resize(cropped_img, (640, 384))
        re_img = re_img.astype(np.uint8)
        re_img = cv2.cvtColor(re_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_image_path, re_img)

if __name__ == "__main__": 

    import numpy as np
    import tqdm
    import os
    import cv2
    from multiprocessing.pool import ThreadPool



    #timestamps_path = '/mnt/nas/madhu/robotcar/night/2014-12-16-18-44-24/files/2014-12-16-18-44-24_train_1m.txt'
    #timestamps_path = '/mnt/nas/madhu/awsgpu2/datasets/robotcar/2014-12-09-13-21-02/2014-12-09-13-21-02_train_1m.txt'
    #timestamps_path = '/home/madhu/code/feature-slam/git_repos/dtd/notebooks/unique_paired_night_day.txt'
    #timestamps_path = '/home/madhu/code/feature-slam/git_repos/matching_night_day_test_split.txt'
    #timestamps_path = '/home/madhu/code/feature-slam/git_repos/temporal_dtd/pose_eval/timestamps.txt'
    timestamps_path = '/mnt/nas/madhu/robotcar/night/2014-12-16-18-44-24/files/2014-12-16-18-44-24_pose_test.txt'
    models_dir = '/mnt/nas/madhu/data/robotcar/models/'
    raw_image_dir = '/hdd1/madhu/data/robotcar/2014-12-16-18-44-24/stereo/right/'
    save_image_dir = '/hdd1/madhu/data/robotcar/2014-12-16-18-44-24/stereo/right_rgb/data/'
    camera_model = CameraModel(models_dir, raw_image_dir)
    data = np.loadtxt(timestamps_path, dtype = int)
    #data = data[:,1]
    data = np.unique(data)


    if not os.path.exists(save_image_dir):
        os.makedirs(save_image_dir)

    try:
        with ThreadPool() as pool:
            
            for _ in tqdm.tqdm(pool.imap_unordered(read_and_save, data), total=len(data)):
                pass

            
            #for timestamp in tqdm.tqdm(data, total = len(data)):
            #    read_and_save(timestamp)
        print('All images in train set are converted to RGB and saved in {}'.format(save_image_dir))
    except Exception as e:
        print(e)
        breakpoint()
    

