import torch
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np
from sent_sampling.utils.optim_utils import optim_pool
from sent_sampling.utils.data_utils import SENTENCE_CONFIG, COCA_PREPROCESSED_DIR, SAVE_DIR,save_obj,load_obj, construct_stimuli_set_from_pd
from sent_sampling.utils import extract_pool

if __name__ == '__main__':
    extract_id = 'group=best_performing_pereira_1-dataset=coca_preprocessed_all_clean_no_dup_100K_sample_1_textNoPeriod-activation-bench=None-ave=False'
    ext_obj = extract_pool[extract_id]()
    ext_obj.load_dataset()
    ext_obj()
    suffix='split_0'

    for idx, model_id in enumerate(ext_obj.model_spec):
        model_activation_name = f"{ext_obj.dataset}_{ext_obj.stim_type}_{ext_obj.model_spec[idx]}_layer_{ext_obj.layer_spec[idx]}_{ext_obj.extract_name}_ave_{ext_obj.average_sentence}.pkl"
        model_activation_name_suffix = f"{ext_obj.dataset}_{suffix}_{ext_obj.stim_type}_{ext_obj.model_spec[idx]}_layer_{ext_obj.layer_spec[idx]}_{ext_obj.extract_name}_ave_{ext_obj.average_sentence}.pkl"
        # see whether model activation already extracted
        if os.path.exists(os.path.join(SAVE_DIR, model_activation_name)):
            model_activation = load_obj(os.path.join(SAVE_DIR, model_activation_name))
            # create a split
            model_activation_split=model_activation[0:50000]
            # save the split
            # print what is being saved
            print(f"Saving {model_activation_name_suffix}")
            save_obj(model_activation_split, os.path.join(SAVE_DIR, model_activation_name_suffix))

    # get sentences
    sent_ids=[x[2] for x in model_activation_split]
    # go ot ext_obj.data_ and get the rows that have the same sent_id as sent_ids and put it ina data_split
    data_split=ext_obj.data_[ext_obj.data_['sent_id'].isin(sent_ids)]
    # get ext_obj.datafile  and dleete .pkl and add suffix + .pkl
    new_data_file=ext_obj.datafile.replace('.pkl',f'_{suffix}.pkl')
    # save the data_split to new_data_file
    save_obj(data_split,new_data_file)

    #construct_stimuli_set_from_pd(data_split,f"{ext_obj.dataset}_{suffix}",drop_period=True,splits=20)
    #stimuli_set = construct_stimuli_set_from_pd(data_split, f"{ext_obj.dataset}_{suffix}", drop_period=True, splits=20)


    suffix = 'split_0'
    extract_id = f'group=best_performing_pereira_1-dataset=coca_preprocessed_all_clean_no_dup_100K_sample_1_{suffix}_textNoPeriod-activation-bench=None-ave=False'
    ext_obj = extract_pool[extract_id]()
    #ext_obj.load_dataset()
    ext_obj()

    optim_id = 'coordinate_ascent_eh-obj=2-D_s-n_iter=500-n_samples=200-n_init=1-low_dim=False-pca_var=0.9-pca_type=pytorch-run_gpu=True'
    optim_obj = optim_pool[optim_id]()

    optim_obj.load_extractor(ext_obj)
    optim_obj.extractor_obj.N_S = 50000
    low_resolution= False
    cpu_dump=True
    preload=False
    save_results=True
    optim_obj.device='cuda'
    self=optim_obj
    optim_obj.precompute_corr_rdm_on_gpu(low_resolution=low_resolution, cpu_dump=cpu_dump, preload=preload,
                                                 save_results=save_results)


    matrix_size_in_bytes = XY_corr.element_size() * XY_corr.nelement()
    print(f"Matrix size: {matrix_size_in_bytes / (1024**2)} MB")

    device1 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device2 = torch.device('cuda:1' if torch.cuda.device_count() > 1 else 'cpu')
    devices = [device1, device2]
    self.XY_corr_list = []
    xy_dir = os.path.join(SAVE_DIR,
                          f"{self.extractor_obj.identifier}_XY_corr_list-low_res={low_resolution}-low_dim={self.low_dim}-pca_type={self.pca_type}-pca_var={self.pca_var}.pkl")
    if not cpu_dump:
        target_device = self.device
    else:
        target_device = torch.device('cpu')
    if low_resolution:
        dtype = torch.float16
    else:
        dtype = torch.float32
    self.XY_corr_list = []
    for idx, act_dict in tqdm(enumerate(self.activations)):
        # backward compatiblity
        True
        act_ = [x[0] if isinstance(act_dict['activations'][0], list) else x for x in act_dict['activations']]
        if idx % 2==0:
            act = torch.tensor(np.asarray(act_), dtype=dtype, device=devices[0], requires_grad=False)
        elif idx % 2 != 0:
            act = torch.tensor(np.asarray(act_), dtype=dtype, device=devices[-1], requires_grad=False)
        XY_corr = corrcoef_metric(act)
        self.XY_corr_list.append(XY_corr.to(target_device))
        del act
        del act_
        del XY_corr
        torch.cuda.empty_cache()

    xy_dir = os.path.join(SAVE_DIR,
                          f"{self.extractor_obj.identifier}_XY_corr_list-low_res={low_resolution}-low_dim={self.low_dim}-pca_type={self.pca_type}-pca_var={self.pca_var}.pkl")

    xy_pt = os.path.join(SAVE_DIR,
                          f"{self.extractor_obj.identifier}_XY_corr_list-low_res={low_resolution}-low_dim={self.low_dim}-pca_type={self.pca_type}-pca_var={self.pca_var}.pt")

    D_precompute = self.XY_corr_list
    save_obj(D_precompute, xy_dir)
    torch.save(D_precompute, xy_pt)