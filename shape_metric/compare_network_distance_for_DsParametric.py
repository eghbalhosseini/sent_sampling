from netrep.metrics import LinearMetric
from pathlib import Path
import pickle as pkl
from sklearn.model_selection import cross_validate
from netrep.multiset import pairwise_distances, frechet_mean
import itertools
import numpy as np
from tqdm import tqdm
import matplotlib
#matplotlib.rcParams.update({'font.family': 'Helvetica', 'font.size': 3,'font.weight':'normal'})
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = False
from scipy.spatial.distance import pdist
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
'''ANN result across models'''
model_layers = [('roberta-base', 'encoder.layer.1'),
                ('xlnet-large-cased', 'encoder.layer.23'),
                ('bert-large-uncased-whole-word-masking', 'encoder.layer.11.output'),
                ('xlm-mlm-en-2048', 'encoder.layer_norm2.11'),
                ('gpt2-xl', 'encoder.h.43'),
                ('albert-xxlarge-v2', 'encoder.albert_layer_groups.4'),
                ('ctrl', 'h.46'),]

import multiprocessing
import os
import seaborn as sns
print(f'num cpus: {multiprocessing.cpu_count()}')
# set omp threads to 1 to avoid slowdowns due to parallelization
os.environ['OMP_NUM_THREADS'] = '4'
import matplotlib.pyplot as plt
if __name__ == '__main__':
    # compute the simliarty vs score
    #%%
    # load act_leftout
    act_dir='/Users/eghbalhosseini/MyData/neural_nlp_bench/activations/DsParametricfMRI/'
    model_resp_leftout=[]
    model_resp_dsparametric=[]
    model_resp_all=[]
    for model_,layer in model_layers:
        model_name = model_
        save_path = Path(f'{act_dir}/act_leftout_dsparametric_{model_name}.pkl')
        # make sure parent exist
        # load from save path
        with open(save_path, 'rb') as f:
            sent_end_layer = pkl.load(f)
        model_resp_leftout.append(sent_end_layer)
        save_path = Path(f'{act_dir}/act_all_dsparametric_{model_name}.pkl')
        # make sure parent exist
        # load from save path
        with open(save_path, 'rb') as f:
            sent_end_layer = pkl.load(f)
        model_resp_all.append(sent_end_layer['act_all'])
        ds_min_ids = sent_end_layer['min_loc']
        ds_max_ids = sent_end_layer['max_loc']
        ds_rand_ids = sent_end_layer['rand_loc']
        all_sentences= sent_end_layer['sentences']


        save_path = Path(f'{act_dir}/act_dict_dsparametric_{model_name}.pkl')
        # make sure parent exist
        with open(save_path, 'rb') as f:
            sent_end_layer = pkl.load(f)
        model_resp_dsparametric.append(sent_end_layer)


    #%%
    feature_map_min=[]
    feature_map_max=[]
    feature_map_rand=[]
    feature_map_leftout=[]
    feature_map_all=[]
    feature_map_joint=[]
    all_var_explained=[]
    for idx in range(len(model_resp_dsparametric)):
        pca=PCA(n_components=500)
        pca.fit(model_resp_leftout[idx])
        var_explained = pca.explained_variance_ratio_
        all_var_explained.append(var_explained)
        combined_act=np.concatenate([model_resp_dsparametric[idx]['act_min'],model_resp_dsparametric[idx]['act_rand'],model_resp_dsparametric[idx]['act_max']])
        assert(model_layers[idx][0]==model_resp_dsparametric[idx]['model_name'])
        feature_map_min.append(pca.transform(model_resp_dsparametric[idx]['act_min']))
        feature_map_max.append(pca.transform(model_resp_dsparametric[idx]['act_max']))
        feature_map_rand.append(pca.transform(model_resp_dsparametric[idx]['act_rand']))
        feature_map_leftout.append(pca.transform(model_resp_leftout[idx]))
        feature_map_joint.append(pca.transform(combined_act))
        feature_map_all.append(pca.transform(model_resp_all[idx]))
    model_group_act = {'min': feature_map_min, 'rand': feature_map_rand, 'max': feature_map_max, 'leftout': feature_map_leftout}

    #%% perform mulitset distance
    grp = 'orth'  # or 'perm' or 'identity' , 'orth' is the default
    method = 'streaming'  # or 'streaming' , 'full_batch' is the default
    adjust_mode = 'none'  # 'pca' or 'none' or 'zero_pad'
    tolerance = 1e-5
    verbose = True
    save_path = Path(f'{act_dir}/multi_shape_distance_DsParametric_{grp}_{adjust_mode}_{method}.pkl')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        with open(save_path, 'rb') as f:
            multi_set_alignment = pkl.load(f)
    else:
        multi_set_alignment = dict()
        for idx, (stim_group, model_act) in enumerate(model_group_act.items()):
            X = [act_ for act_ in model_act]
            if adjust_mode == 'zero_pad':
                X_shape = [x.shape[-1] for x in X]
                max_shape = max(X_shape)
                # pad each X with zeros to make it max_shape
                X_pad = [np.pad(x, ((0, 0), (0, max_shape - x.shape[-1])), 'constant') for x in X]
            elif adjust_mode == 'none':
                X_pad = X

            X_var, aligned_Xs = frechet_mean(X_pad, group=grp, method=method, return_aligned_Xs=True,max_iter=200, verbose=verbose,
                                             tol=tolerance)
            multi_set_alignment[stim_group] = [aligned_Xs, X_var]
            # pdist(np.stack(aligned_Xs).reshape(len(X), -1))
        # # make sure parent exist

        # # save the data as a pickle file
        with open(save_path, 'wb') as f:
             pkl.dump(multi_set_alignment, f)
    # load the distmats

    alinged_Xs_min, X_var_min = multi_set_alignment['min']
    dist_min = pdist(np.stack(alinged_Xs_min).reshape(len(alinged_Xs_min), -1))
    alinged_Xs_rand, X_var_rand = multi_set_alignment['rand']
    dist_random = pdist(np.stack(alinged_Xs_rand).reshape(len(alinged_Xs_rand), -1))
    alinged_Xs_max, X_var_max = multi_set_alignment['max']
    dist_max = pdist(np.stack(alinged_Xs_max).reshape(len(alinged_Xs_max), -1))
    # do a pca on [X_var_min,X_var_rand, X_var_max]
    pca = PCA(n_components=2)
    # X_var_min_rand_max= np.concatenate([X_var_min,X_var_rand, X_var_max], axis=0)
    # X_var_input=X_var_rand
    # pca.fit(X_var_input)
    # X_var_pca = pca.transform(X_var_min_rand_max)
    #
    # # # shw the variance explained
    # pca.explained_variance_ratio_
    # # # split the pca into min and max and rand
    # #
    # X_var_min= X_var_pca[:len(X_var_min), :]
    # X_var_rand=X_var_pca[len(X_var_min):len(X_var_min)+len(X_var_max), :]
    # X_var_max=X_var_pca[len(X_var_min)+len(X_var_max):, :]


    X_var_pca = pca.fit_transform(np.concatenate([X_var_min, X_var_max], axis=0))
    pca.explained_variance_ratio_

    X_var_min= X_var_pca[:len(X_var_min), :]
    X_var_max=X_var_pca[len(X_var_min):, :]

    #%%
    sorted_image_ids=np.stack([np.arange(model_resp_dsparametric[0]['sent_min'].shape[0]), np.arange(model_resp_dsparametric[0]['sent_max'].shape[0])]).flatten()

    sent_min=list(model_resp_dsparametric[0]['sent_min'])
    sent_max=list(model_resp_dsparametric[0]['sent_max'])
    sent_=np.stack([sent_min, sent_max]).flatten()
    # select sentence min that corresponds to index 0
    list(model_resp_dsparametric[0]['sent_min'])[72]

    list(model_resp_dsparametric[0]['sent_max'])[65]

    sent_max=model_resp_dsparametric[0]['sent_max']

    x_pca = np.concatenate((X_var_min[:, :2], X_var_max[:, :2]), axis=0)
    # create labels max and min
    labels = np.concatenate((np.repeat('min', X_var_min.shape[0]), np.repeat('max', X_var_min.shape[0])), axis=0)
    # create a df with x_pca and labels
    df = pd.DataFrame(x_pca, columns=['x', 'y'])
    df['labels'] = labels
    df['image_id'] = sorted_image_ids
    df['sent'] = sent_
    # save df as a mat file using scipy
    # from scipy.io import savemat
    # save_path = Path(f'{act_dir}/multi_shape_distance_DsParametric_pca_min_max_{grp}_{adjust_mode}_{method}.mat')
    # # make sure parent exist
    # save_path.parent.mkdir(parents=True, exist_ok=True)
    # # save as a mat file
    # savemat(save_path, {'x': x_pca, 'labels': labels, 'image_id': sorted_image_ids, 'sent': sent_})

    # Define your color palette for groups
    color_palette = {'max': np.divide((0, 157, 255, 255), 255), 'min': np.divide((255, 98, 0, 255), 255)}

    # Initialize a JointGrid
    g = sns.JointGrid(data=df, x="x", y="y")
    # Plot each group on the same JointGrid
    for group, color in color_palette.items():
        sns.scatterplot(data=df[df['labels'] == group], x="x", y="y", color=color, ax=g.ax_joint)
    # plot a horizontal line at origin
    g.ax_joint.axhline(y=0, color='gray', linestyle='--')
    g.ax_joint.axvline(x=0, color='gray', linestyle='--')
    # add the image_id as a text next to the point
    # Plot the marginals
    sns.histplot(data=df, x="x", hue="labels", palette=color_palette, ax=g.ax_marg_x, legend=False, binwidth=20,
                 element="step", fill=False)
    sns.histplot(data=df, y="y", hue="labels", palette=color_palette, ax=g.ax_marg_y, legend=False, binwidth=20,
                 element="step", fill=False)
    g.savefig(os.path.join(act_dir, f'DsParametricfMRI_Align_pca_min_max_{grp}_{adjust_mode}_{method}.png'))
    # save eps
    g.savefig(os.path.join(act_dir, f'DsParametricfMRI_Align_pca_min_max_{grp}_{adjust_mode}_{method}.eps'),
              format='eps')
    #%%
    save_path = Path(
        f'{act_dir}/multi_shape_distance_all_DsParametric_{grp}_{adjust_mode}_{method}.pkl')
    if save_path.exists():
        with open(save_path, 'rb') as f:
            all_set_alignment = pkl.load(f)
    else:
        X = feature_map_all
        all_set_alignment = dict()
        if adjust_mode == 'zero_pad':
            X_shape = [x.shape[-1] for x in X]
            max_shape = max(X_shape)
            # pad each X with zeros to make it max_shape
            X_pad = [np.pad(x, ((0, 0), (0, max_shape - x.shape[-1])), 'constant') for x in X]
        elif adjust_mode == 'none':
            X_pad = X

        X_var, aligned_Xs = frechet_mean(X_pad, group=grp, method=method, return_aligned_Xs=True, max_iter=200,
                                         verbose=verbose, tol=tolerance)
        all_set_alignment['all'] = [aligned_Xs, X_var]
        # save the data as a pickle file
        with open(save_path, 'wb') as f:
            pkl.dump(all_set_alignment, f)

    # make sure parent exist

    aligned_Xs, X_var = all_set_alignment['all']
    pca = PCA(n_components=2)
    X_var_pca = pca.fit_transform(X_var)
    # get var explained
    pca.explained_variance_ratio_
    fig, ax = plt.subplots()
    ax.scatter(X_var_pca[:, 0], X_var_pca[:, 1], label='min')
    plt.show()


    X_var_min=    X_var_pca[sorted(ds_min_ids), :]
    X_var_max=    X_var_pca[sorted(ds_max_ids), :]
    X_var_rand=    X_var_pca[sorted(ds_rand_ids), :]
    sent_min=[all_sentences[i] for i in ds_min_ids]
    sent_max=[all_sentences[i] for i in ds_max_ids]
    sent_rand=[all_sentences[i] for i in ds_rand_ids]
    # make a flat list of sent_min and sent_max
    selected_image_paths = list(itertools.chain.from_iterable(zip(sent_min, sent_max)))

    x_pca = np.concatenate((X_var_min[:,:2],X_var_max[:,:2] ), axis=0)
    # create labels max and min
    labels = np.concatenate((np.repeat('min', X_var_min.shape[0]), np.repeat('max', X_var_min.shape[0])), axis=0)
    # create a df with x_pca and labels

    df = pd.DataFrame(x_pca, columns=['x', 'y'])
    df['labels'] = labels
    df['sentences']=selected_image_paths
    # Define your color palette for groups
    color_palette = {'max':np.divide((0, 157, 255,255), 255), 'min': np.divide((255, 98, 0,255), 255)}

    fig = plt.figure(figsize=(8, 11))
    fig.dpi = 500
    # fig_length = 0.055 * len(models_scores)
    pap_ratio = 8 / 11
    ax = plt.axes((.04, .04, .6, .6 * pap_ratio))
    # Plot each group on the same JointGrid
    for group, color in color_palette.items():
        sns.scatterplot(data=df[df['labels'] == group], x="x", y="y", color=color, ax=ax)
    # plot a horizontal line at origin
    # ax.axhline(y=0, color='gray', linestyle='-')
    # ax.axvline(x=0, color='gray', linestyle='-')
    # add the image_id as a text next to the point
    for idx in range(80):
        row = df.iloc[idx]
        # ab = AnnotationBbox(getImage(row['image_path'],zoom=0.025), (row['x'], row['y']), frameon=False)
        # create a textbox with the sentence, and if it is too long, split it into two lines
        sent = row['sentences']
        sent = sent.split(' ')
        if len(sent) > 10:
            sent = ' '.join(sent[:10]) + '\n' + ' '.join(sent[10:])
        else:
            sent = ' '.join(sent)
        ax.text(row['x'], row['y'], sent, horizontalalignment='left', color='black', fontsize=3)

    ax = plt.axes((.04, .52, .6, .6 * pap_ratio))
    # Plot each group on the same JointGrid
    for group, color in color_palette.items():
        sns.scatterplot(data=df[df['labels'] == group], x="x", y="y", color=color, ax=ax)
    # plot a horizontal line at origin
    # ax.axhline(y=0, color='gray', linestyle='-')
    # ax.axvline(x=0, color='gray', linestyle='-')
    # add the image_id as a text next to the point
    for idx in range(80, 160):
        row = df.iloc[idx]
        # ab = AnnotationBbox(getImage(row['image_path'],zoom=0.025), (row['x'], row['y']), frameon=False)
        # create a textbox with the sentence, and if it is too long, split it into two lines
        sent = row['sentences']
        sent = sent.split(' ')
        if len(sent) > 10:
            sent = ' '.join(sent[:10]) + '\n' + ' '.join(sent[10:])
        else:
            sent = ' '.join(sent)
        ax.text(row['x'], row['y'], sent, horizontalalignment='left', color='black', fontsize=3)

    fig.savefig(os.path.join(act_dir, f'DsParametricfMRI_Align_all_pca_min_max_{grp}_{adjust_mode}_{method}.png'))
    # save eps
    fig.savefig(os.path.join(act_dir, f'DsParametricfMRI_Align_all_pca_min_max_{grp}_{adjust_mode}_{method}.eps'),
              format='eps')

    #%% do the optimization on comined min,rand and max data
    X = feature_map_joint
    all_set_alignment = dict()
    if adjust_mode == 'zero_pad':
        X_shape = [x.shape[-1] for x in X]
        max_shape = max(X_shape)
        # pad each X with zeros to make it max_shape
        X_pad = [np.pad(x, ((0, 0), (0, max_shape - x.shape[-1])), 'constant') for x in X]
    elif adjust_mode == 'none':
        X_pad = X

    X_var, aligned_Xs = frechet_mean(X_pad, group=grp, method=method, return_aligned_Xs=True, max_iter=200,
                                     verbose=verbose, tol=tolerance)
    all_set_alignment['joint'] = [aligned_Xs, X_var]



    pca = PCA(n_components=2)
    X_var_pca = pca.fit_transform(X_var)
    # split the pca into min and max and rand
    X_var_min=    X_var_pca[:len(X_var_min), :]
    X_var_rand=    X_var_pca[len(X_var_min):len(X_var_min)+len(X_var_max), :]
    X_var_max=    X_var_pca[len(X_var_min)+len(X_var_max):, :]

    x_pca = np.concatenate((X_var_min[:, :2], X_var_max[:, :2]), axis=0)
    # create labels max and min
    labels = np.concatenate((np.repeat('min', X_var_min.shape[0]), np.repeat('max', X_var_min.shape[0])), axis=0)
    # create a df with x_pca and labels
    df = pd.DataFrame(x_pca, columns=['x', 'y'])
    df['labels'] = labels
    df['image_id'] = sorted_image_ids
    df['sent'] = sent_
    # Define your color palette for groups
    color_palette = {'max': np.divide((0, 157, 255, 255), 255), 'min': np.divide((255, 98, 0, 255), 255)}

    # Initialize a JointGrid
    g = sns.JointGrid(data=df, x="x", y="y")

    # Plot each group on the same JointGrid
    for group, color in color_palette.items():
        sns.scatterplot(data=df[df['labels'] == group], x="x", y="y", color=color, ax=g.ax_joint)
    # plot a horizontal line at origin
    g.ax_joint.axhline(y=0, color='gray', linestyle='--')
    g.ax_joint.axvline(x=0, color='gray', linestyle='--')
    for idx in range(80):
        row = df.iloc[idx]
        g.ax_joint.text(row['x'], row['y'], row['sent'], horizontalalignment='left', color='black', fontsize=3)
    # Plot the marginals

    g.fig.show()