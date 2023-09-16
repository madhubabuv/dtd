import numpy as np
import matplotlib.pyplot as plt


def comparison_bar_plot(mono,stereo):

    print(mono[2:].mean(0))
    print(stereo[2:].mean(0))

    print(mono.shape)
    print(stereo.shape)

    #plt.figure(figsize=(6,4))

    bar_width = 1.5
    x = np.arange(0,55)[::5]

    plt.bar(x, mono[:,1], bar_width)
    plt.bar(x + bar_width, stereo[:,1], bar_width)
    plt.xlabel('Depth buckets (m)',fontsize=18)
    plt.grid()
    plt.tight_layout()
    plt.ylabel('Sq. Rel Error',fontsize=18)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.legend(['Ours','IGEV-Stereo'],fontsize=18)
    

    # ax[0].bar(x , mono[1:,2], bar_width, label='DINO')
    # #ax[0].bar(x, stereo[1:,2][::space], bar_width, label='UniMatch')
    # #ax[0].bar(x + bar_width, mono_stereo[1:,2][::space], bar_width, label='mono_stereo')
    # ax[0].set_ylabel('RMSE')
    # ax[0].legend()
    # ax[0].grid()

    # ax[1].bar(x - bar_width, mono[1:,-3][::space], bar_width, label='DINO')
    # ax[1].bar(x, stereo[1:,-3][::space], bar_width, label='UniMatch')
    # #ax[1].bar(x + bar_width, mono_stereo[1:,-3][::space], bar_width, label='mono_stereo')
    # ax[1].set_xlabel('Depth buckets')
    # ax[1].set_ylabel('delta1')
    # ax[1].legend()
    # ax[1].grid()
    plt.tight_layout()
    plt.savefig('ours_vs_igev.png')



def weighted_mean(data_path):
    data = np.load(data_path)
    weights = data != 999
    weights = weights.astype(np.float32)
    data = weights * data
    data = np.sum(data,axis=0)
    weights = np.sum(weights,axis=0)
    data = data / weights

    return data

# I am going to plot two main metrics: RMSE and delta1

data2 = weighted_mean('bin_wise_metrics_MS2_sgm.npy')
data1 = weighted_mean('bin_wise_metrics_MS2_ours.npy')

# data3 = weighted_mean('bin_wise_metrics_mono_stereo_monodepth2.npy')

comparison_bar_plot(data1,data2)



'''
data = weighted_mean('bin_wise_metrics.npy')
plt.grid()
fig, ax = plt.subplots(2,1)

ax[0].bar(np.arange(1,80,2)[1:],data[1:,2], label='RMSE',align='edge',width=1.0)
ax[0].set_ylabel('RMSE')
ax[0].grid()

ax[1].bar(np.arange(1,80,2)[1:],data[1:,-3], label='delta1',align='edge',width=1.0)
ax[1].set_ylabel('delta1')
ax[1].set_xlabel('Depth buckets')
ax[1].grid()
plt.tight_layout() 
plt.savefig('metrics.png')
'''