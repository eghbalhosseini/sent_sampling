import sys
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
from sent_sampling.utils import extract_pool
from sent_sampling.utils.optim_utils import optim_pool
import argparse
from sent_sampling.utils.data_utils import RESULTS_DIR, save_obj,SAVE_DIR,load_obj
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='extract activations and optimize')
parser.add_argument('extractor_id', type=str,
                    default='group=set_3-dateset=ud_sentences_filter-network_act-bench=None-ave=False')
parser.add_argument('optimizer_id', type=str, default='coordinate_ascent-obj=D_s-n_iter=100-n_samples=100-n_init=1')
import numpy as np
args = parser.parse_args()

if __name__ == '__main__':
    #extractor_id = args.extractor_id
    #optimizer_id = args.optimizer_id

    extractor_id = f'group=gpt2-xl_layers-dataset=coca_spok_filter_punct_50K_sylb-activation-bench=None-ave=False'
    optimizer_id = f"coordinate_ascent_eh-obj=D_s-n_iter=1000-n_samples=250-n_init=1-run_gpu=True"
    low_resolution = 'False'
    low_dim = 'False'
    print(extractor_id + '\n')
    print(optimizer_id + '\n')
    # extract data
    extractor_obj = extract_pool[extractor_id]()
    extractor_obj.load_dataset()
    extractor_obj()
    # optimize
    optimizer_obj = optim_pool[optimizer_id]()
    optimizer_obj.load_extractor(extractor_obj)
    xy_dir = os.path.join(SAVE_DIR,
                          f"{optimizer_obj.extractor_obj.identifier}_XY_corr_list-low_res={low_resolution}_low_dim={low_dim}.pkl")

    if os.path.exists(xy_dir):
        print(f'loading precomputed correlation matrix from {xy_dir}')
        D_precompute = load_obj(xy_dir)
        optimizer_obj.XY_corr_list = D_precompute
    else:
        # optimizer_obj.N_S=
        # save_obj(self.XY_corr_list,xy_dir)
        print('precomputing correlation matrix ')
        optimizer_obj.precompute_corr_rdm_on_gpu(low_resolution=False, cpu_dump=True, preload=False, save_results=True)

    # preloaded optimized sentences :



    optimized_set=[13197,13215,3869,4883,4735,15016,8315,11384,12880,1630,5318,18768,815,18616,139,4224,9761,8369,5803,14394,13797,296,1865,8046,18237,4185,
                   195,18428,12479,7648,5335,18891,9538,7225,989,10223,12997,1189,4997,5595,6626,5643,12316,13116,1149,14725,14295,4668,7002,8960,1501,907,
                   4620,4000,5722,19379,7587,13539,9684,9852,3300,19037,3166,77,7576,7096,19609,10891,5845,2467,10338,298,9112,5258,10705,3200,4391,11234,18359,
                   7694,17200,8871,11995,13354,1562,12184,16785,4722,8829,13272,4126,1892,1163,7914,12145,14169,12409,15190,13300,13945,5600,2840,14461,8406,1210,
                   12002,2894,10536,15143,16838,7355,1383,9018,9132,12490,1404,3307,8797,2677,5260,6146,7783,10333,17001,10853,929,3527,18211,14051,11185,9523,1290,
                   14467,19205,392,15214,11737,12886,2601,7136,12497,14396,9693,10260,3765,13141,12450,16277,7684,1031,9416,18071,14124,17252,17083,2526,3365,2615,
                   2325,12427,10953,2633,17073,360,2699,7779,7774,7890,4657,19035,4346,11972,4758,14928,17094,16835,17069,9711,3863,16085,4123,17440,1548,5395,15360,
                   14355,15380,6294,17122,17723,18857,348,4018,731,4328,5487,9866,12231,8017,3474,8368,12687,11968,9820,2706,3382,817,16004,13792,4468,15768,12120,14071,
                   9869,8133,14863,16392,11882,9986,11402,15528,661,6409,15153,11509,5459,5364,18398,11931,8466,3217,3305,19776,5055,4453,13160,13873,9382,16460,18025,3944,
                   15508,10650,12619,4638,2768,12606,16205,5771,7058]


    optimizer_obj.gpu_object_function(optimized_set)
    (a,b)=optimizer_obj.gpu_object_function_debug(optimized_set)

    a_rnd=[]
    b_rnd=[]
    for kk in tqdm(range(200)):
        sent_random = list(np.random.choice(optimizer_obj.N_S, optimizer_obj.N_s))
        (a, b) = optimizer_obj.gpu_object_function_debug(sent_random)
        a_rnd.append(a)
        b_rnd.append(b)
    a_rnd=np.asarray(a_rnd)
    plt.imshow(b.numpy(),vmin=0,vmax=1.6)
    plt.colorbar()
    plt.show()
    np.mean(a_rnd)
    b_rnd_mean=np.stack(b_rnd).mean(axis=0)
    plt.imshow(b_rnd_mean,vmax=1.6)
    plt.colorbar()
    plt.show()
    #
    ax = plt.axes((.1, .1, .05, .35))

    ax.scatter(.2 * np.random.normal(size=(a_rnd.shape)) , a_rnd, color=(.6, .6, .6),
               s=2, alpha=.3)
    ax.scatter(0, a_rnd.mean(), color=(0, 0, 0), s=50, label='random')
    ax.scatter(0, a, color=(1, 0, 0), s=50, label='optimized')
    # ax.scatter(idx,res['optimized_d'],height=0.4,color=alph_col[[idx],:],alpha=.9,edgecolor=(0,0,0),linewidth=2,label=res['optimizatin_name'])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.set_xlim((-.4, .4))
    ax.set_ylim((0, 1))
    ax.set_xticks([])
    ax.tick_params(direction='out', length=3, width=2, colors='k',
                   grid_color='k', grid_alpha=0.5)

    ax.legend(bbox_to_anchor=(5.1, .85), frameon=True)
    plt.show()
    # sentences

    values = []
    optimizer_obj.extractor_obj
    values=[( optimizer_obj.extractor_obj.data_[id]['text']) for id in optimized_set]
    with open(os.path.join('/om/user/ehoseini/MyData/sent_sampling/coca_spok_filter_punct_50K/text/', f"sentences_gpt2_xl_layers_iter_7_{optimizer_id}.txt"), 'w') as f:
                for item in values:
                    f.write("%s\n" % (item))



    optim_sentences=[(optimizer_obj.extractor_obj.data_[id]['text']) for id in optimized_set]
    sent_lengths=[len(x.split()) for x in optim_sentences]
    (x,y)=np.histogram(sent_lengths,bins=np.arange(1,20))
    len(x)
    len(y)
    plt.hist(sent_lengths,bins=np.arange(5,18))
    plt.show()
