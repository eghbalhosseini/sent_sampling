import sys
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
from utils import extract_pool
from utils.optim_utils import optim_pool
import argparse
from utils.data_utils import RESULTS_DIR, save_obj,SAVE_DIR,load_obj
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

    extractor_id = f'group=gpt2-xl_layers-dataset=coca_spok_filter_punct_50K_sylb_2to4sec-activation-bench=None-ave=False'
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



    optimized_set=[5224,468,7425,8906,6147,2577,6947,3267,4957,11512,7963,6551,6813,10720,8219,4664,9697,10081,9584,618,
                   2023,1465,5881,8708,3386,9667,7584,4687,5689,2452,4291,407,3314,4277,6968,7476,1126,5849,7328,287,2692,
                   6645,9362,5242,195,7508,6028,3578,3827,3464,10590,7967,103,10296,3880,5303,5252,8739,10200,6808,2579,
                   10754,11018,11599,3288,11547,6117,2164,6675,6447,10647,7591,824,11242,5007,356,4726,2043,6894,4151,8191,
                   8539,7305,4048,7315,7503,6280,2534,11201,9039,2388,4096,4515,4429,846,9112,3034,11631,3326,10856,3215,
                   1435,835,9899,11185,11543,8632,5773,7302,3099,11318,4103,5617,3526,7620,6056,9237,1058,8501,2089,10061,
                   10581,8531,3233,2249,6040,1940,5396,7731,10108,5759,7360,217,1335,10030,4477,9680,4581,6148,10649,5721,
                   8380,11021,10215,2275,1076,7431,814,9639,7300,2913,8795,2519,194,4184,5840,6356,246,4268,5917,8839,1134,
                   1883,9606,5413,3361,5401,1429,3025,2744,5340,11198,6617,4891,7057,7072,5890,3615,8256,1431,928,8461,9196,
                   769,6778,9480,7210,10410,7817,6736,6936,4113,9120,3894,230,7956,9957,8854,881,216,7908,11551,8026,4274,
                   1910,3124,1691,7952,8266,10140,7347,5599,5006,1576,74,1619,6727,8755,1918,10473,7928,1485,2040,9991,7069,
                   8460,10706,4630,3453,705,389,5563,2034,5744,5746,8437,369,8184,3942,1939,4436,6797,920,2305,2780,4717,
                   2386,7613,10093,8888]


    optimizer_obj.gpu_object_function(optimized_set)
    (a,b)=optimizer_obj.gpu_object_function_debug(optimized_set)

    a_rnd=[]
    b_rnd=[]
    for kk in tqdm(range(200)):
        sent_random = list(np.random.choice(optimizer_obj.N_S, optimizer_obj.N_s))
        (a_r, b_r) = optimizer_obj.gpu_object_function_debug(sent_random)
        a_rnd.append(a_r)
        b_rnd.append(b_r.cpu().numpy())
    a_rnd=np.asarray(a_rnd)
    fig = plt.figure(figsize=(11, 8), dpi=100, frameon=False)
    ax = plt.axes((.05, .1, .4, .4))

    plt.imshow(b.cpu().numpy(),vmin=0,vmax=1.6)
    ax.set_title('optimized')
    ax.set_xlabel('layer')
    ax.set_ylabel('layer')

    plt.colorbar()


    ax = plt.axes((.5, .1, .4, .4))
    b_rnd_mean=np.stack(b_rnd).mean(axis=0)
    plt.imshow(b_rnd_mean,vmax=1.6)
    ax.set_title('random')
    plt.colorbar()

    fig.show()
    #
    fig = plt.figure(figsize=(11, 8), dpi=100, frameon=False)
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

    ax.legend(bbox_to_anchor=(3.1, .85), frameon=True)
    fig.show()
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
