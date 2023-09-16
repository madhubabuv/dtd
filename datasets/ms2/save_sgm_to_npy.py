import numpy as np
import os
import cv2
import tqdm
if __name__ == "__main__":

    data = '/hdd1/madhu/data/ms2/test_nighttime_list.txt'
    data = np.loadtxt(data, dtype=str, delimiter=',')
    left_file_names = data[:,0]
    right_file_names = data[:,1]

    sgm_filenames = []

    sgm_path = '/mnt/nas/madhu/data/predictions/baslines/MS2_sgm/'

    for name in left_file_names:

        split_name = name.split('/')
        image_idx = split_name[-1].split('.')[0]
        seq_id = split_name[-4]
        file_path = os.path.join(sgm_path, seq_id+'_'+image_idx+'.png')
        assert os.path.exists(file_path), file_path
        sgm_filenames.append(file_path)

    all_disps = []
    for file_name in tqdm.tqdm(sgm_filenames, total = len(sgm_filenames)):

        img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
        #center crop
        height, width = img.shape[0], img.shape[1]
        new_height = 384
        new_width = 640

        left = (width - new_width)//2
        top = (height - new_height)//2
        right = (width + new_width)//2
        bottom = (height + new_height)//2

        img = img[top:bottom, left:right]

        all_disps.append(img)

    all_disps = np.stack(all_disps, axis=0)
    np.save('/mnt/nas/madhu/data/predictions/baslines/MS2_sgm.npy', all_disps)



