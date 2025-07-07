from netrep.metrics import LinearMetric
from pathlib import Path
import pickle as pkl
from sklearn.model_selection import cross_validate
from netrep.multiset import pairwise_distances, frechet_mean
import itertools
import numpy as np
from tqdm import tqdm

from scipy.spatial.distance import pdist
import numpy as np
from sklearn.decomposition import PCA
from glob import glob
import pickle
import matplotlib
matplotlib.rcParams.update({'font.family': 'Helvetica', 'font.size': 7,'font.weight':'normal'})
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = False
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
def getImageZoom(path,zoom=0.1):
    return OffsetImage(plt.imread(path), zoom=zoom)


def getImage(path, ax):
    img = Image.open(path)
    # Resize image to fit within the plot better. Adjust as necessary.
    img = img.resize((int(ax.figure.dpi/4), int(ax.figure.dpi/4)))
    return OffsetImage(np.array(img))
def pca_on_individual_matrices(matrices, target_size):
    """
    Perform PCA separately on each matrix in the list.
    """
    pca = PCA(n_components=target_size)
    transformed_matrices = []

    for matrix in matrices:
        pca.fit(matrix)
        transformed_matrix = pca.transform(matrix)
        transformed_matrices.append(transformed_matrix)

    return transformed_matrices

import multiprocessing
import os
from sent_sampling.utils import make_shorthand
print(f'num cpus: {multiprocessing.cpu_count()}')
# set omp threads to 1 to avoid slowdowns due to parallelization
os.environ['OMP_NUM_THREADS'] = '4'
import pandas as pd
import seaborn as sns
if __name__ == '__main__':
    # compute the simliarty vs score
    extract_mode='redux'
    n_samples=80
    optim_id_min = f'coordinate_ascent_eh-obj=2-D_s-n_iter=500-n_samples={n_samples}-n_init=1-low_dim=False-pca_var=0.9-pca_type=sklearn-run_gpu=True'
    optim_id_max = f'coordinate_ascent_eh-obj=D_s-n_iter=500-n_samples={n_samples}-n_init=1-low_dim=False-pca_var=0.9-pca_type=sklearn-run_gpu=True'
    optim_id_rand = f'coordinate_ascent_eh-obj=D_s_rand-n_iter=500-n_samples={n_samples}-n_init=1-low_dim=False-pca_var=0.9-pca_type=sklearn-run_gpu=True'
    deepjuice_id = f'group=deepjuice_models-dataset=nsd-{extract_mode}-bench=None-ave=False'
    selected_models = ['torchvision_alexnet_imagenet1k_v1',
                       'torchvision_regnet_x_800mf_imagenet1k_v2',
                       'openclip_vit_b_32_laion2b_e16',
                       'timm_swinv2_cr_tiny_ns_224',
                       'torchvision_efficientnet_b1_imagenet1k_v2',
                       'clip_rn50',
                       'timm_convnext_large_in22k',
                       ]
    models_sh = ['AlexNet', 'RegNet', 'ViT', 'Swin', 'EfficientNet', 'CLIP', 'ConvNext']
    image_paths='/Users/eghbalhosseini/MyData/DeepJuice/NSD_image_paths.pkl'
    # read image path
    with open(image_paths, 'rb') as f:
        image_paths = pickle.load(f)
    activations_list = []
    layers_list = []
    # for to deepjuice path and find model activation in the format
    deepjuice_ws_path = '/Users/eghbalhosseini/MyData/DeepJuice/workspace'
    ANN_activation_paret = '/Users/eghbalhosseini/MyData/neural_nlp_bench/activations/'
    for model_ in selected_models:
        save_file = f'{deepjuice_ws_path}/nsd/{model_}*{extract_mode}.pkl'
        original_files = glob(save_file)
        # open the file
        with open(original_files[0], 'rb') as f:
            original = pickle.load(f)
        layer_id = original[0]
        act_ = original[1]
        activation = dict(model_name=model_, layer=layer_id, activations=act_)
        activations_list.append(activation)
        layers_list.append(layer_id)

    feature_maps = [x['activations'] for x in activations_list]
    results_path = '/Users/eghbalhosseini/MyData/DeepJuice/sampling'
    (extract_short_hand, optim_short_hand_min) = make_shorthand(deepjuice_id, optim_id_min)
    ds_min_path = f'{results_path}/results_{extract_short_hand}_{optim_short_hand_min}_{extract_mode}.pkl'
    with open(ds_min_path, 'rb') as f:
        results_ds_min = pickle.load(f)

    (extract_short_hand, optim_short_hand_max) = make_shorthand(deepjuice_id, optim_id_max)
    ds_max_path = f'{results_path}/results_{extract_short_hand}_{optim_short_hand_max}_{extract_mode}.pkl'
    with open(ds_max_path, 'rb') as f:
        results_ds_max = pickle.load(f)

    (extract_short_hand, optim_short_rand) = make_shorthand(deepjuice_id, optim_id_rand)
    ds_rand_path = f'{results_path}/results_{extract_short_hand}_{optim_short_rand}_{extract_mode}.pkl'
    with open(ds_rand_path, 'rb') as f:
        results_ds_rand = pickle.load(f)
    ds_min_image_ids = results_ds_min['optimized_S']
    ds_max_image_ids = results_ds_max['optimized_S']
    ds_rand_image_ids = results_ds_rand['optimized_S']
    feature_map_min = [x[sorted(ds_min_image_ids), :] for x in feature_maps]
    feature_map_max = [x[sorted(ds_max_image_ids), :] for x in feature_maps]
    feature_map_rand = [x[sorted(ds_rand_image_ids), :] for x in feature_maps]
    sorted_image_ids = np.stack([sorted(ds_min_image_ids), sorted(ds_max_image_ids)]).flatten()
    selected_image_paths = [image_paths[x] for x in sorted_image_ids]
    # strip the path to the image name
    selected_image_names = [x.split('/')[-1] for x in selected_image_paths]
    # drop the extension
    selected_image_names = [x.split('.')[0] for x in selected_image_names]

    # put them together
    # create a lofout feature set
    d_id_leftout = list(
        set(np.arange(0, 1000)) - set(ds_min_image_ids) - set(ds_max_image_ids) - set(ds_rand_image_ids))
    assert len(set(d_id_leftout).intersection(set(ds_min_image_ids))) == 0
    assert len(set(d_id_leftout).intersection(set(ds_max_image_ids))) == 0
    assert len(set(d_id_leftout).intersection(set(ds_rand_image_ids))) == 0
    feature_map_leftout = [x[sorted(d_id_leftout), :] for x in feature_maps]
    # for eahc model do a pca on feature_map_leftout and apply it to feature_map_min, feature_map_max, feature_map_rand
    all_var_explained = []
    feature_map_all_=[]
    for idx in range(len(feature_maps)):
        pca = PCA(n_components=500)
        pca.fit(feature_map_leftout[idx])
        # compute variance explained
        var_explained = pca.explained_variance_ratio_
        all_var_explained.append(var_explained)
        feature_map_min[idx] = pca.transform(feature_map_min[idx])
        feature_map_max[idx] = pca.transform(feature_map_max[idx])
        feature_map_rand[idx] = pca.transform(feature_map_rand[idx])
        feature_map_all_.append(pca.transform(feature_maps[idx]))
    # create a model_group_act dictionary
    # compute the sum of variance explained
    all_var_explained = np.stack(all_var_explained)
    sum_var_explained = np.sum(all_var_explained, axis=1)
    model_group_act = {'min':feature_map_min,'rand': feature_map_rand,'max': feature_map_max}


    ## perform mulitset distance
    grp='orth' # or 'perm' or 'identity'
    method='full_batch' # or 'streaming'
    adjust_mode='none' # 'pca' or 'none'
    tolerance=1e-5
    verbose=True
    multi_set_alignment = dict()
    for idx,(stim_group,model_act) in enumerate(model_group_act.items()):
        X=[act_ for act_ in model_act]
        if adjust_mode=='zero_pad':
            X_shape = [x.shape[-1] for x in X]
            max_shape = max(X_shape)
            # pad each X with zeros to make it max_shape
            X_pad = [np.pad(x, ((0, 0), (0, max_shape - x.shape[-1])), 'constant') for x in X]
        elif adjust_mode=='none':
            X_pad = X

        X_var, aligned_Xs = frechet_mean(X_pad,group=grp,method=method, return_aligned_Xs=True,max_iter=200,verbose=verbose,tol=tolerance)
        multi_set_alignment[stim_group] = [aligned_Xs, X_var]
        #pdist(np.stack(aligned_Xs).reshape(len(X), -1))
    # save the distmats
    save_path = Path(f'{ANN_activation_paret}/{deepjuice_id}/multi_shape_distance_DsParametric_DeepJuice_{grp}_{adjust_mode}_{method}.pkl')
    # make sure parent exist
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # save the data as a pickle file
    with open(save_path, 'wb') as f:
        pkl.dump(multi_set_alignment, f)
    # load the distmats
    with open(save_path, 'rb') as f:
        multi_set_alignment = pkl.load(f)

    alinged_Xs_min, X_var_min = multi_set_alignment['min']
    dist_min=pdist(np.stack(alinged_Xs_min).reshape(len(alinged_Xs_min), -1))
    alinged_Xs_rand, X_var_rand = multi_set_alignment['rand']
    dist_random=pdist(np.stack(alinged_Xs_rand).reshape(len(alinged_Xs_rand), -1))
    alinged_Xs_max, X_var_max = multi_set_alignment['max']

    pca = PCA(n_components=2)
    X_var_pca = pca.fit_transform(np.concatenate([X_var_min, X_var_rand, X_var_max], axis=0))
    # split the pca into min and max and rand
    # show variance explained for pca fit
    0

    X_var_min = X_var_pca[:len(X_var_min), :]
    X_var_rand = X_var_pca[len(X_var_min):len(X_var_min) + len(X_var_max), :]
    X_var_max = X_var_pca[len(X_var_min) + len(X_var_max):, :]
    # X_var_pca = pca.fit_transform(np.concatenate([X_var_min, X_var_max], axis=0))
    # # # split the pca into min and max and rand
    # X_var_min = X_var_pca[:len(X_var_min), :]
    # # X_var_rand = X_var_pca[len(X_var_min):len(X_var_min) + len(X_var_max), :]
    # X_var_max = X_var_pca[len(X_var_min):, :]



    x_pca = np.concatenate((X_var_min[:,:2],X_var_max[:,:2] ), axis=0)
    # create labels max and min
    labels = np.concatenate((np.repeat('min', X_var_min.shape[0]), np.repeat('max', X_var_min.shape[0])), axis=0)
    # create a df with x_pca and labels
    colors = np.concatenate([np.tile(np.divide((0, 157, 255,255), 255),(X_var_min.shape[0],1)), np.tile(np.divide((255, 98, 0,255), 255),(X_var_min.shape[0],1))],axis=0)
    df = pd.DataFrame(x_pca, columns=['x', 'y'])
    df['labels'] = labels
    df['image_id'] = selected_image_names
    # Define your color palette for groups
    color_palette = {'max':np.divide((0, 157, 255,255), 255), 'min': np.divide((255, 98, 0,255), 255)}

    # Initialize a JointGrid
    g = sns.JointGrid(data=df, x="x", y="y")

    # Plot each group on the same JointGrid
    for group, color in color_palette.items():
        sns.scatterplot(data=df[df['labels'] == group], x="x", y="y", color=color, ax=g.ax_joint)
    # plot a horizontal line at origin
    g.ax_joint.axhline(y=0, color='gray', linestyle='--')
    g.ax_joint.axvline(x=0, color='gray', linestyle='--')
    # add the image_id as a text next to the point
    # for idx in range(80):
    #     row = df.iloc[idx]
    #     g.ax_joint.text(row['x'], row['y'], row['image_id'], horizontalalignment='left', size='small', color='black')

    # Plot the marginals
    sns.histplot(data=df, x="x", hue="labels", palette=color_palette, ax=g.ax_marg_x, legend=False,binwidth=20,element="step", fill=False)
    sns.histplot(data=df, y="y", hue="labels", palette=color_palette, ax=g.ax_marg_y, legend=False,binwidth=20,element="step", fill=False)


    # Show the plot

    g.savefig(os.path.join(ANN_activation_paret, f'{extract_short_hand}_ds_min_ds_max_pca_{grp}_{adjust_mode}_{method}.png'))
    #save eps
    g.savefig(os.path.join(ANN_activation_paret, f'{extract_short_hand}_ds_min_ds_pca_{grp}_{adjust_mode}_{method}.eps'),
           format='eps')


    X = feature_map_all_
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
    X_var.shape
    save_path = Path(f'{ANN_activation_paret}/{deepjuice_id}/multi_shape_distance_all_DsParametric_DeepJuice_{grp}_{adjust_mode}_{method}.pkl')
    # make sure parent exist
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # save the data as a pickle file
    with open(save_path, 'wb') as f:
        pkl.dump(all_set_alignment, f)

    # do a pca on X_var
    pca = PCA(n_components=2)
    X_var_pca = pca.fit_transform(X_var)

    fig, ax = plt.subplots()
    ax.scatter(X_var_pca[:, 0], X_var_pca[:, 1], label='min')
    plt.show()
    X_var_min=    X_var_pca[sorted(ds_min_image_ids), :]
    X_var_max=    X_var_pca[sorted(ds_max_image_ids), :]
    X_var_rand=    X_var_pca[sorted(ds_rand_image_ids), :]



    x_pca = np.concatenate((X_var_min[:,:2],X_var_max[:,:2] ), axis=0)
    # create labels max and min
    labels = np.concatenate((np.repeat('min', X_var_min.shape[0]), np.repeat('max', X_var_min.shape[0])), axis=0)
    # create a df with x_pca and labels
    colors = np.concatenate([np.tile(np.divide((0, 157, 255,255), 255),(X_var_min.shape[0],1)), np.tile(np.divide((255, 98, 0,255), 255),(X_var_min.shape[0],1))],axis=0)
    df = pd.DataFrame(x_pca, columns=['x', 'y'])
    df['labels'] = labels
    df['image_id'] = selected_image_names
    df['image_path']=selected_image_paths
    # Define your color palette for groups
    color_palette = {'max':np.divide((0, 157, 255,255), 255), 'min': np.divide((255, 98, 0,255), 255)}

    # Initialize a JointGrid
    fig = plt.figure(figsize=(8, 11))
    fig.dpi = 500
    # fig_length = 0.055 * len(models_scores)
    pap_ratio= 8 / 11
    ax = plt.axes((.04, .04, .6, .6 * pap_ratio))
    # Plot each group on the same JointGrid
    for group, color in color_palette.items():
        sns.scatterplot(data=df[df['labels'] == group], x="x", y="y", color=color, ax=ax)
    # plot a horizontal line at origin
    #ax.axhline(y=0, color='gray', linestyle='-')
    #ax.axvline(x=0, color='gray', linestyle='-')
    # add the image_id as a text next to the point
    for idx in range(80):
        row = df.iloc[idx]
        #ab = AnnotationBbox(getImage(row['image_path'],zoom=0.025), (row['x'], row['y']), frameon=False)
        ab = AnnotationBbox(getImage(row['image_path'], ax), (row['x'], row['y']), frameon=False)
        ax.add_artist(ab)
        #g.ax_joint.text(row['x'], row['y'], row['image_id'], horizontalalignment='left', size='small', color='black')


    ax = plt.axes((.04, .52, .6, .6 * pap_ratio))
    # Plot each group on the same JointGrid
    for group, color in color_palette.items():
        sns.scatterplot(data=df[df['labels'] == group], x="x", y="y", color=color, ax=ax)
    # plot a horizontal line at origin
    # ax.axhline(y=0, color='gray', linestyle='-')
    # ax.axvline(x=0, color='gray', linestyle='-')
    # add the image_id as a text next to the point
    for idx in range(80,160):
        row = df.iloc[idx]
        # ab = AnnotationBbox(getImage(row['image_path'],zoom=0.025), (row['x'], row['y']), frameon=False)
        ab = AnnotationBbox(getImage(row['image_path'], ax), (row['x'], row['y']), frameon=False)
        ax.add_artist(ab)
        # g.ax_joint.text(row['x'], row['y'], row['image_id'], horizontalalignment='left', size='small', color='black')

    fig.savefig(os.path.join(ANN_activation_paret, f'{extract_short_hand}_Align_all_pca_min_max_{grp}_{adjust_mode}_{method}_images.png'))
    #save eps
    fig.savefig(os.path.join(ANN_activation_paret, f'{extract_short_hand}_Align_all_pca_min_max_{grp}_{adjust_mode}_{method}_images.eps'),
           format='eps')



    # create labels max and min

    # create a df with x_pca and labels
    df = pd.DataFrame(X_var_pca, columns=['x', 'y'])
    df['image_path']= image_paths
    # Define your color palette for groups
    # Initialize a JointGrid
    fig = plt.figure(figsize=(8, 11))
    fig.dpi = 500
    # fig_length = 0.055 * len(models_scores)
    pap_ratio= 8 / 11
    ax = plt.axes((.04, .04, .8, .8 * pap_ratio))
    # Plot each group on the same JointGrid

    sns.scatterplot(data=df, x="x", y="y", ax=ax)
    # plot a horizontal line at origin
    #ax.axhline(y=0, color='gray', linestyle='-')
    #ax.axvline(x=0, color='gray', linestyle='-')
    # add the image_id as a text next to the point
    for idx in range(df.shape[0]):
        row = df.iloc[idx]
        #ab = AnnotationBbox(getImage(row['image_path'],zoom=0.025), (row['x'], row['y']), frameon=False)
        ab = AnnotationBbox(getImage(row['image_path'], ax), (row['x'], row['y']), frameon=False)
        ax.add_artist(ab)
        #g.ax_joint.text(row['x'], row['y'], row['image_id'], horizontalalignment='left', size='small', color='black')


    fig.savefig(os.path.join(ANN_activation_paret, f'{extract_short_hand}_Align_all_pca_all_{grp}_{adjust_mode}_{method}_images.png'))
    #save eps
    fig.savefig(os.path.join(ANN_activation_paret, f'{extract_short_hand}_Align_all_pca_all_{grp}_{adjust_mode}_{method}_images.eps'),
           format='eps')


    for idm, model_name in enumerate(models_sh):
        x_pca = np.concatenate((feature_map_min[idm][:, :2], feature_map_max[idm][:, :2]), axis=0)
        # create labels max and min
        labels = np.concatenate((np.repeat('min', feature_map_min[idm].shape[0]), np.repeat('max', feature_map_min[idm].shape[0])), axis=0)
        # create a df with x_pca and labels
        colors = np.concatenate([np.tile(np.divide((0, 157, 255, 255), 255), (feature_map_min[idm].shape[0], 1)),
                                 np.tile(np.divide((255, 98, 0, 255), 255), (feature_map_min[idm].shape[0], 1))], axis=0)
        df = pd.DataFrame(x_pca, columns=['x', 'y'])
        df['labels'] = labels
        df['image_id'] = selected_image_names
        df['image_path'] = selected_image_paths
        # Define your color palette for groups
        color_palette = {'max': np.divide((0, 157, 255, 255), 255), 'min': np.divide((255, 98, 0, 255), 255)}

        # Initialize a JointGrid
        fig = plt.figure(figsize=(8, 11))
        fig.dpi = 250
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
            ab = AnnotationBbox(getImage(row['image_path'], ax), (row['x'], row['y']), frameon=False)
            #ab = AnnotationBbox(getImageZoom(row['image_path']), (row['x'], row['y']), frameon=False)
            ax.add_artist(ab)
            # g.ax_joint.text(row['x'], row['y'], row['image_id'], horizontalalignment='left', size='small', color='black')

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
            ab = AnnotationBbox(getImage(row['image_path'], ax), (row['x'], row['y']), frameon=False)
            #ab = AnnotationBbox(getImageZoom(row['image_path'],zoom=72./fig.dpi), (row['x'], row['y']), frameon=False)
            ax.add_artist(ab)
            # g.ax_joint.text(row['x'], row['y'], row['image_id'], horizontalalignment='left', size='small', color='black')

        fig.savefig(os.path.join(ANN_activation_paret,
                                 f'{extract_short_hand}_{model_name}_pca_min_max_images.png'))
        # save eps
        fig.savefig(os.path.join(ANN_activation_paret,
                                 f'{extract_short_hand}_{model_name}_pca_min_max_images.eps'),
                    format='eps')



