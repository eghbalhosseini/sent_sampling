import sys
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
from utils import extract_pool
from utils.optim_utils import optim_pool
import argparse
from utils.data_utils import RESULTS_DIR, save_obj,SAVE_DIR, load_obj
import os
parser = argparse.ArgumentParser(description='extract activations and optimize')
parser.add_argument('extractor_id', type=str,
                    default='group=set_3-dateset=ud_sentences_filter-network_act-bench=None-ave=False')
parser.add_argument('optimizer_id', type=str, default='coordinate_ascent-obj=D_s-n_iter=100-n_samples=100-n_init=1')

args = parser.parse_args()

if __name__ == '__main__':
    #extractor_id = args.extractor_id
    #optimizer_id = args.optimizer_id

    extractor_id = f'group=gpt2-xl_layers-dataset=coca_spok_filter_punct_50K_sylb-activation-bench=None-ave=False'
    optimizer_id = f"coordinate_ascent_eh-obj=D_s-n_iter=1000-n_samples=250-n_init=1-run_gpu=True"
    low_resolution='False'
    low_dim='False'
    print(extractor_id+'\n')
    print(optimizer_id+'\n')
    # extract data
    extractor_obj = extract_pool[extractor_id]()
    extractor_obj.load_dataset()
    extractor_obj()
    # optimize
    optimizer_obj = optim_pool[optimizer_id]()
    optimizer_obj.load_extractor(extractor_obj)
    xy_dir=os.path.join(SAVE_DIR, f"{optimizer_obj.extractor_obj.identifier}_XY_corr_list-low_res={low_resolution}_low_dim={low_dim}.pkl")

    if os.path.exists(xy_dir):
        print(f'loading precomputed correlation matrix from {xy_dir}')
        D_precompute=load_obj(xy_dir)
        optimizer_obj.XY_corr_list=D_precompute
    else:
        #optimizer_obj.N_S=
    # save_obj(self.XY_corr_list,xy_dir)
        print('precomputing correlation matrix ')
        optimizer_obj.precompute_corr_rdm_on_gpu(low_resolution=False,cpu_dump=True,preload=False,save_results=True)
    S_opt_d, DS_opt_d = optimizer_obj()
    # save results
    optim_results = dict(extractor_name=extractor_id,
                         model_spec=extractor_obj.model_spec,
                         layer_spec=extractor_obj.layer_spec,
                         data_type=extractor_obj.extract_type,
                         benchmark=extractor_obj.extract_benchmark,
                         average=extractor_obj.average_sentence,
                         optimizatin_name=optimizer_id,
                         optimized_S=S_opt_d,
                         optimized_d=DS_opt_d)
    optim_file=os.path.join(RESULTS_DIR,f"results_{extractor_id}_{optimizer_id}.pkl")
    save_obj(optim_results, optim_file)


    # test compartmentilation
    # self=optimizer_obj
    # self.XY_corr_list = []
    # cpu_dump=True
    # low_resolution=False
    # if not cpu_dump:
    #     target_device = self.device
    # else:
    #     target_device = torch.device('cpu')
    # for idx, act_dict in tqdm(enumerate(self.activations)):
    #     # backward compatiblity
    #     act_ = [x[0] if isinstance(act_dict['activations'][0], list) else x for x in act_dict['activations']]
    #     act = torch.tensor(act_, dtype=float, device=self.device, requires_grad=False)
    #     X = torch.nn.functional.normalize(act.squeeze())
    #     X = X - X.mean(axis=1, keepdim=True)
    #     X = torch.nn.functional.normalize(X)
    #
    #     if low_resolution == True:
    #         XY_corr = torch.tensor(1, device=self.device, dtype=torch.float16) - torch.mm(X, torch.transpose(X, 1,
    #                                                                                                          0)).half()
    #     else:
    #         XY_corr = torch.tensor(1, device=self.device, dtype=float) - torch.mm(X, torch.transpose(X, 1, 0))
    #
    #     XY_corr= XY_corr.to(target_device)
    #     self.XY_corr_list.append(XY_corr)
    #     del X
    #     del act
    #     del XY_corr
    #     torch.cuda.empty_cache()
    # xy_dir=os.path.join(SAVE_DIR, f"{extractor_id}_XY_corr_list-low_res={low_resolution}.pkl")
    # save_obj(self.XY_corr_list,xy_dir)