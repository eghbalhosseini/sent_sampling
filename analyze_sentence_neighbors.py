import warnings
import numpy as np
import os
os.chdir('/om/user/ehoseini/sent_sampling')
os.getcwd()
import torch.cuda
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
plt.interactive(False)
import matplotlib.cm as cm
import matplotlib

matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import re
from utils import extract_pool , gpt2_xl_grp_config
from utils.extract_utils import model_extractor, model_extractor_parallel, SAVE_DIR
from utils.data_utils import SENTENCE_CONFIG,  RESULTS_DIR, save_obj, load_obj, ANALYZE_DIR
import utils.optim_utils
from utils.optim_utils import optim, optim_pool, pt_create_corr_rdm_short, optim_group
import argparse
device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# rapids stuff
import cudf, cuml
from cuml.neighbors import NearestNeighbors
from cuml import PCA
from cuml.decomposition import PCA
from sklearn.cluster import SpectralClustering
import pandas as pd
from pathlib import Path
torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # first load ev sentenes:
    # first find ev sentences
    file_name = 'U01_sentselection_Dec18-2020_updDec23.xlsx'
    df_ev_selected = pd.read_excel(os.path.join(RESULTS_DIR, f"{file_name}"))
    ev_sentences = df_ev_selected.sentence[df_ev_selected.previously_selected_by_ev == 1]
    # load one extractor
    extractor_name='group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3-activation-bench=None-ave=False'
    extractor_obj = extract_pool[extractor_name]()
    extractor_obj.load_dataset()
    extractor_obj()
    sentences = [x['text'] for x in extractor_obj.data_]
    # find location of ev sentences in the set
    ev_selected_idx=[sentences.index(x) for x in ev_sentences]
    # select a reference model for finind neighbors:
    model_name = 'gpt2-xl'
    model_names = [x['model_name'] for x in extractor_obj.model_group_act]
    model_dat=extractor_obj.model_group_act[model_names.index(model_name)]
    model_act=torch.Tensor([x[0] for x in model_dat['activations']]).to(device)
    # define a distance metric and find neighbors
    d_metric='euclidean'
    knn_mdl = NearestNeighbors(n_neighbors=7, two_pass_precision=True,metric=d_metric)
    knn_mdl.fit(model_act)
    distances, indices = knn_mdl.kneighbors(model_act)
    # find the corresponding sentence for the neighbors
    sent_neighbor = []
    for _,ind in tqdm(enumerate(indices)):
        sent_neighbor.append('\t '.join([sentences[x] for x in list(ind.get())]))
    # save a text file with corresponding sentences:
    # do spectral clustering on knn graph
    textfile = open(f'{RESULTS_DIR}/{Path(file_name).stem}_nearest_neighbors_{model_name}_dist_metric_{d_metric}.txt', "w")
    for idx in ev_selected_idx:
        textfile.write(sent_neighbor[idx] + "\n")
    textfile.close()
    del textfile
    # compare the values for D_s between the first set and second set.
    optim_ids = 'coordinate_ascent_eh-obj=D_s-n_iter=1000-n_samples=200-n_init=1-run_gpu=True'
    optimizer_obj = optim_pool[optim_ids]()
    optimizer_obj.load_extractor(extractor_obj)
    optimizer_obj.precompute_corr_rdm_on_gpu(low_dim=True, low_resolution=True, cpu_dump=True)
    ev_sent_Ds=optimizer_obj.gpu_object_function(ev_selected_idx)

    ev_sent_neighbors=list(indices.get()[ev_selected_idx,6])
    neighbor_Ds=optimizer_obj.gpu_object_function_debug(ev_sent_neighbors)
    # random set
    ds_random=[]
    for _,_ in tqdm(enumerate(range(200))):
        rand_set=list(np.random.choice(optimizer_obj.N_S,size=len(ev_selected_idx),replace=False))
        ds_rnd=optimizer_obj.gpu_object_function(rand_set)
        ds_random.append(ds_rnd)

    # plot the results

    fig = plt.figure(figsize=(14, 7), dpi=100, frameon=False)
    ax = plt.axes((.1, .1, .1, .75))

    cmap = cm.get_cmap('viridis_r')
    tick_l = []
    tick = []
    idx = 0
    D_s_rand = ds_random
    ax.scatter(.2 * np.random.normal(size=(np.asarray(D_s_rand).shape)) + idx, np.asarray(D_s_rand),
               color=(.6, .6, .6),
               s=2, alpha=.2)
    ax.scatter(idx, np.asarray(D_s_rand).mean(), color=(.6, .6, .6), s=20,
               label=f'random, size={optimizer_obj.N_s}')
    ax.scatter(idx, ev_sent_Ds, color=(1, .7, 0), edgecolor=(.2, .2, .2), s=50,
               label=f'optimized')
    ev_neighb_list=indices.get()
    neighb_col = cmap(np.divide(range(ev_neighb_list.shape[1]), ev_neighb_list.shape[1]))
    for idn in range(ev_neighb_list.shape[1]-1):
        ev_sent_neighbors = list(ev_neighb_list[ev_selected_idx, idn+1])
        neighbor_Ds = optimizer_obj.gpu_object_function(ev_sent_neighbors)
        ax.scatter(idx, neighbor_Ds, color=neighb_col[idn+1,:], edgecolor=(.2, .2, .2), s=30,
               label=f'neighbor {idn+1}')
    #
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.set_xlim((-.8, 2.8))
    ax.set_ylim((.7, 1.1))
    ax.set_xticks([])
    ax.tick_params(direction='out', length=3, width=2, colors='k',
                   grid_color='k', grid_alpha=0.5)
    ax.legend(bbox_to_anchor=(2.1, .85), frameon=True)
    ax.set_ylabel(r'$D_s$')
    ax.set_title(f'change in Ds when choosing neighbors in ANN_SET1 \n model: {model_name}, neighborhood metric: {d_metric}',horizontalalignment='left')
    plt.show()
    '''
    optim_ids = 'coordinate_ascent_eh-obj=D_s-n_iter=1000-n_samples=250-n_init=1-run_gpu=True'
    # load a set of sentences

    #results_file = 'results_group=gpt2-xl_layer_compare_v1-dataset=coca_spok_filter_punct_10K_sample_1-activation-bench=None-ave=False_coordinate_ascent_eh-obj=D_s-n_iter=1000-n_samples=200-n_init=1-run_gpu=True.pkl'
    results_file='results_group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3-activation-bench=None-ave=False_coordinate_ascent-obj=D_s-n_iter=10000-n_samples=200-n_init=1-run_gpu=True.pkl'
    if os.path.exists(os.path.join(RESULTS_DIR,results_file)):
        optim_result=load_obj(os.path.join(RESULTS_DIR,results_file))
    optim_result['optimizatin_name']
    optim_result['extractor_name']
    selected_sent=optim_result['optimized_S']
    # create an optim object
    extractor_obj = extract_pool[optim_result['extractor_name']]()
    extractor_obj.load_dataset()
    extractor_obj()
    # get sentences
    sentences=[x['text'] for x in extractor_obj.data_]
    optimizer_obj = optim_pool[optim_result['optimizatin_name']]()
    optimizer_obj.load_extractor(extractor_obj)

    model_name='gpt2-xl'
    model_names=[x['model_name'] for x in optimizer_obj.extractor_obj.model_group_act]

    # model
    layer_dat=optimizer_obj.extractor_obj.model_group_act[model_names.index(model_name)]
    # compute the rdm
    optimizer_obj.precompute_corr_rdm_on_gpu(low_dim=True, low_resolution=True, cpu_dump=True)
    optimizer_obj.gpu_object_function(selected_sent)

    layer_act=torch.Tensor([x[0] for x in layer_dat['activations']]).to(device)
    layer_similarity=optimizer_obj.XY_corr_list[model_names.index(model_name)]

    #del optimizer_obj
    del extractor_obj

    layer_similarity=layer_similarity.to(device)

    [topk_val,topk_idx]=torch.topk(layer_similarity, 5, dim=1, largest=False, sorted=True)
    # create an index matrix
    sent_neighbor=[]
    [sent_neighbor.append('\t '.join([sentences[x] for x in topk_idx[k, :]] )) for k in range(topk_idx.shape[0])]
    opt_res=optim_result['extractor_name']
    textfile = open(f'{RESULTS_DIR}/{opt_res}_nearest_neighbors_{model_name}.txt', "w")
    for idx in selected_sent:
        textfile.write(sent_neighbor[idx] + "\n")

    # find neighbors

    knn_m = NearestNeighbors(n_neighbors=7,two_pass_precision=True)
    X_cudf = cudf.DataFrame(layer_act)
    knn_m.fit(layer_act)


    distances, indices = knn_m.kneighbors(X_cudf)
    distances, indices = knn_m.kneighbors(layer_act)
    sent_neighbor = []
    [sent_neighbor.append('\t '.join([sentences[x] for x in list(neighbors.get())])) for neighbors in indices]


    # do spectral clustering on knn graph
    textfile = open(f'{RESULTS_DIR}/{opt_res}_nearest_neighbors_{model_name}_euclidean.txt', "w")

    for idx in selected_sent:
        textfile.write(sent_neighbor[idx] + "\n")
    textfile.close()
    del textfile

    # do a version based on mean responses
    optim_result['extractor_name']
    mean_ext_name=optim_result['extractor_name'].replace('ave=False','ave=True')
    # create an optim object
    extractor_obj = extract_pool[mean_ext_name]()
    extractor_obj.load_dataset()
    extractor_obj()
    sentences = [x['text'] for x in extractor_obj.data_]
    model_name = 'gpt2-xl'
    model_names = [x['model_name'] for x in optimizer_obj.extractor_obj.model_group_act]


    optimizer_obj = optim_pool[optim_result['optimizatin_name']]()
    optimizer_obj.load_extractor(extractor_obj)
    layer_dat = optimizer_obj.extractor_obj.model_group_act[model_names.index(model_name)]
    layer_act = torch.Tensor([x[0] for x in layer_dat['activations']]).to(device)
    layer_act.shape
    pca_=PCA(n_components=250)
    pca_.fit(layer_act)
    print(f'explained variance: {pca_.explained_variance_ratio_}')
    plt.plot(np.cumsum(pca_.explained_variance_ratio_.get()[:100]))
    plt.show()
    pca_ld=PCA(n_components=np.where(np.cumsum(pca_.explained_variance_ratio_.get()[:100])>.8)[0][0])
    layer_act_ld=pca_ld.fit_transform(layer_act)
    # do neighbor analysis on the data
    knn_m = NearestNeighbors(n_neighbors=7, two_pass_precision=True)
    knn_m.fit(layer_act_ld)
    distances, indices = knn_m.kneighbors(layer_act_ld)
    a=knn_m.kneighbors_graph(layer_act_ld)
    sent_neighbor = []
    for _,ind in tqdm(enumerate(indices)):
        sent_neighbor.append('\t '.join([sentences[x] for x in list(ind.get())]))


    # do spectral clustering on knn graph

    textfile = open(f'{RESULTS_DIR}/{mean_ext_name}_nearest_neighbors_{model_name}_euclidean_low_dim.txt', "w")

    for idx in selected_sent:
        textfile.write(sent_neighbor[idx] + "\n")
    textfile.close()
    del textfile
    # try umap as well
    umap=cuml.UMAP(n_neighbors=10,n_components=2)
    umap.fit(layer_act_ld)
    embeddings=umap.embedding_.get()
    plt.scatter(embeddings[:,0],embeddings[:,1],marker='.',edgecolors='none')
    plt.scatter(embeddings[selected_sent, 0],embeddings[selected_sent, 1],marker='o',c='k')
    plt.show()

    # try tSNE
    tsne=cuml.TSNE()
    tsne.fit(layer_act_ld)
    embeddings = tsne.embedding_.get()
    plt.scatter(embeddings[:, 0], embeddings[:, 1], marker='.', edgecolors='none')
    plt.scatter(embeddings[selected_sent, 0],embeddings[selected_sent, 1],marker='o',c='k')
    plt.show()
    pc_embd=layer_act_ld.get()
    plt.scatter(pc_embd[:, 0], pc_embd[:, 1], marker='.', edgecolors='none')
    plt.scatter(pc_embd[selected_sent, 0], pc_embd[selected_sent, 1], marker='o', c='k')
    plt.show()

    clustering = SpectralClustering(n_clusters=20,random_state = 0,affinity='precomputed').fit(a.get())

    clustering.labels'''





