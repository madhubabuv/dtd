import numpy as np
import pdb
import tqdm
import matplotlib.pyplot as plt

#gt_depth_path = '/mnt/nas/madhu/data/KITTI/gt_depths/gt_depths.npz'
#depth_data = np.load(gt_depth_path, fix_imports=True, encoding='latin1',allow_pickle=True)["data"]

gt_depth_data = '/hdd1/madhu/data/robotcar/2014-12-16-18-44-24/depth_evaluation/gt_depths.npy'
#gt_depth_data = '/hdd1/madhu/data/ms2/gt_test_depths_filtered.npy'

depth_data = np.load(gt_depth_data,allow_pickle=True)

hist_bins = np.arange(1, 100, 2)
bins_data = []
for depth_map in tqdm.tqdm(depth_data,total = len(depth_data)):
    hist_data = np.histogram(depth_map, bins=hist_bins)[0]
    bins_data.append(hist_data)

bins_data = np.array(bins_data)
bins_data = bins_data.sum(axis=0)

fig, ax = plt.subplots()

ax.bar(hist_bins[:-1], bins_data, width = 1.0,align='edge')
ax.grid()
#plt.bar(hist_bins[:-1], bins_data, width = 0.5)
plt.xlabel('Depth buckets (m)',fontsize = 18)
plt.ylabel('Number of points',fontsize = 18)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.tight_layout()
plt.savefig('MS2_num_of_points_gtdepth_hist.png')

'''




