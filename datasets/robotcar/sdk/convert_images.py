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



if __name__ == "__main__": 

    import numpy as np
    import tqdm
    import os
    import cv2



    #timestamps_path = '/mnt/nas/madhu/robotcar/night/2014-12-16-18-44-24/files/2014-12-16-18-44-24_train_1m.txt'
    timestamps_path = '/mnt/nas/madhu/awsgpu2/datasets/robotcar/2014-12-09-13-21-02/2014-12-09-13-21-02_train_1m.txt'

    models_dir = '/mnt/nas/madhu/data/robotcar/models/'
    raw_image_dir = '/hdd1/madhu/data/robotcar/2014-12-09-13-21-02/stereo/right/'
    save_image_dir = '/hdd1/madhu/data/robotcar/2014-12-09-13-21-02/stereo/right_rgb/data/'
    camera_model = CameraModel(models_dir, raw_image_dir)
    
    data = np.loadtxt(timestamps_path, dtype = int)
    # if len(data.shape) == 2:
    #     data = data[:,0]
    unique_timestamps = np.unique(data)
    #np.random.shuffle(unique_timestamps)
    if not os.path.exists(save_image_dir):
        os.makedirs(save_image_dir)

    try:
        
        for timestamp in tqdm.tqdm(unique_timestamps, total = len(unique_timestamps)):
            filename = os.path.join(raw_image_dir, str(timestamp) + '.png')
            img = load_image(filename, camera_model)
            cropped_img = img[:768, :,:]
            re_img = cv2.resize(cropped_img, (640, 384))
            re_img = re_img.astype(np.uint8)
            re_img = cv2.cvtColor(re_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_image_dir, str(timestamp) + '.png'), re_img)

        print('All images in train set are converted to RGB and saved in {}'.format(save_image_dir))
    except Exception as e:
        print(e)
        breakpoint()
    

