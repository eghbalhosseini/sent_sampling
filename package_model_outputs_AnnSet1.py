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
        ds_ev = optimizer_obj.mod_objective_function([1,2])
        ds_ev = optimizer_obj.mod_objective_function(np.asarray(ev_sent_id))
        result[f"filtering by Ev"] = ds_ev
        #
        # get the random set
        ds_rand = []
        for k in tqdm(enumerate(range(50))):
            sent_random = list(np.random.choice(optimizer_obj.N_S, optimizer_obj.N_s))
            ds_rand.append(optimizer_obj.mod_objective_function(sent_random))
        result[f"random"] = ds_rand