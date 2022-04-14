import os
import numpy as np
from utils.data_utils import load_obj, SAVE_DIR, UD_PARENT, RESULTS_DIR, LEX_PATH_SET, save_obj
from utils import extract_pool
from utils.optim_utils import optim_pool
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import seaborn
from tqdm import tqdm
import pandas as pd

if __name__ == '__main__':
    file_name = 'U01_sentselection_Dec18-2020_updDec23.xlsx'
    df_ev_selected = pd.read_excel(os.path.join(RESULTS_DIR, f"{file_name}"))

    ev_sentences = df_ev_selected.sentence[df_ev_selected.previously_selected_by_ev == 1]

    ds_transform = [
        dict(src='group=best_performing_pereira_1-dataset=ud_sentencez_token_filter_v3-activation-bench=None-ave=False',
             optim='coordinate_ascent-obj=D_s-n_iter=2000-n_samples=300-n_init=2-run_gpu=False')]

    for idx, ds_t in enumerate(ds_transform):
        result = dict()
        optim_file = os.path.join(RESULTS_DIR,
                                  f"results_{ds_t['src']}_{ds_t['optim'].replace('-run_gpu=False', '')}.pkl")
        res = load_obj(optim_file)
        ext_obj = extract_pool[res['extractor_name']]()
        ext_obj.load_dataset()
        ext_obj()
        optimizer_obj = optim_pool[ds_t['optim']]()
        optimizer_obj.load_extractor(ext_obj)
        ds_src = optimizer_obj.mod_objective_function(res['optimized_S'])
        result[f"originial"] = ds_src
        sentences = [x['text'] for x in ext_obj.data_]
        ev_sent_id = ([sentences.index(ev_sent) for ev_sent in ev_sentences])
        ds_ev = optimizer_obj.mod_objective_function(np.asarray(ev_sent_id))
        result[f"filtering by Ev"] = ds_ev
        #
        # get the random set
        ds_rand = []
        for k in tqdm(enumerate(range(50))):
            sent_random = list(np.random.choice(optimizer_obj.N_S, optimizer_obj.N_s))
            ds_rand.append(optimizer_obj.mod_objective_function(sent_random))
        result[f"random"] = ds_rand

    fig = plt.figure(figsize=[15, 8])
    ax = fig.add_axes([.1, .1, .4, .6])

    cmap = cm.get_cmap('viridis_r')

    alph_col = cmap(np.divide(range(len(result)), len(result)))
    tick_l = []

    ax.barh(0, result['originial'], color=alph_col[[0], :], label='originial')
    str_val = "{:.5f}".format(result['originial'])
    tick_l.append(f"original\n {str_val}")

    ax.barh(1, result['filtering by Ev'], color=alph_col[[1], :], label='filtering by Ev')
    str_val = "{:.5f}".format(result['filtering by Ev'])
    tick_l.append(f"filtering by Ev \n {str_val}")

    ax.barh(2, np.mean(result['random']), xerr=np.std(result['random']), color=alph_col[[2], :], label='random')
    str_val = "{:.5f}".format(np.mean(result['random']))
    tick_l.append(f"random \n {str_val}")

    ax.invert_yaxis()

    ax.set_xlabel('D_s')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(tick_l, fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"sentence from {file_name}")
    fig.show()
    #
    model_input = optimizer_obj.extractor_obj.data_
    selected_sentences=[model_input[x] for x in ev_sent_id]
    model_activations=[]
    model_ids=[]
    for k in tqdm(range(len(optimizer_obj.activations))):

        model_act=optimizer_obj.activations[k]['activations']
        act_ = [x[0] if isinstance(model_act[0], list) else x for x in model_act]
        model_activations.append([act_[x] for x in ev_sent_id])
        model_ids.append(optimizer_obj.activations[k]['model_name'])

    # save activation
    act_ev_output = dict(model_names=model_ids,sentences=selected_sentences ,model_acts=model_activations)

    save_obj(act_ev_output,os.path.join(SAVE_DIR,'results','act_ev_AnnSet1.pkl'))
    # save rdms

    rdm_src = optimizer_obj.mod_rdm_function(res['optimized_S'],vector=False)
    model_names=[x['model_name'] for x in optimizer_obj.activations]
    rdm_output=dict(model_names=model_names,rdm_1st=rdm_src['RDM_1st'],rdm_2nd=rdm_src['RDM_2nd'])

    rdm_ev = optimizer_obj.mod_rdm_function(np.asarray(ev_sent_id),vector=False)
    rdm_ev_output = dict(model_names=model_names,sentences=ev_sentences ,rdm_1st=rdm_ev['RDM_1st'], rdm_2nd=rdm_ev['RDM_2nd'])

    save_obj(rdm_ev_output,os.path.join(SAVE_DIR,'results','rdm_ev_AnnSet1.pkl'))

