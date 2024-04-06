import sys
import numpy as np
from scipy.stats import entropy
import logging
import getpass
import itertools
import copy
import xarray as xr
from sklearn.decomposition import PCA
import torch
from tqdm import tqdm
from sent_sampling.utils import extract_pool
device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
import os
import torch.nn.functional as F
from sent_sampling.utils.data_utils import save_obj, SAVE_DIR,load_obj
import torch.distributions as dst
try :
    torch.set_deterministic(True)
except:
    pass
torch.set_printoptions(precision=10)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if getpass.getuser()=='eghbalhosseini':
    OPTIM_PARENT='/Users/eghbalhosseini/MyCodes/opt-exp-design-nlp/'
elif getpass.getuser()=='ehoseini':
    OPTIM_PARENT = '/om/user/ehoseini/opt-exp-design-nlp'
def low_dim_project(act,var_explained=0.90,pca_type='pytorch'):
    # act must be in m sample * n feature shape,
    if pca_type=='pytorch':
        q = min(4000, min(act.shape))
        u, s, v = torch.pca_lowrank(act, q=q)
        var_explained_curve=torch.cumsum(s ** 2, dim=0) / torch.sum(s ** 2)
        idx_cutoff = torch.logical_not(var_explained_curve > var_explained)
        num_dimensions=torch.sum(idx_cutoff)
        s_project=torch.multiply(s,idx_cutoff)
        #act_project=torch.matmul(torch.matmul(u,torch.diag(s_project)),v.transpose(1,0))
        # create a low dimensional representation of the data using u s v and the number of dimensions
        act_project=torch.matmul(u[:,idx_cutoff],torch.matmul(torch.diag(s[idx_cutoff]),v[:, idx_cutoff].transpose(1,0)))
        print(f'pytorch: svd {num_dimensions} dims, vs actual {act.shape[1]}')
    elif pca_type=='sklearn':
        pca=PCA(n_components=var_explained,svd_solver='full')
        device =act.device
        pca.fit(act.cpu().numpy())
        act_project = pca.transform(act.cpu().numpy())
        act_project=torch.from_numpy(act_project).to(device)
        var_explained_curve=pca.explained_variance_ratio_
        print(f'sklearn : pca {pca.explained_variance_ratio_.shape[0]} dims, vs actual {act.shape[1]}')
    return act_project,var_explained_curve

def corrcoef_metric(act):
    """corrcoef"""
    metric_val = 1 - torch.corrcoef(act)
    return metric_val


LOG_BASE = np.e
EPSILON = 1e-16
PRECISION = 1e-200
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s - %(funcName)s] %(message)s')
ch.setFormatter(formatter)


sys.path.insert(1,OPTIM_PARENT)
from opt_exp_design import coordinate_ascent, coordinate_ascent_eh,coordinate_ascent_parallel_eh
import tools
from tools import second_order_rdm, create_rdm, MI
LOGGER = tools.get_logger('OPT-EXP-DSGN')

# Objective functions :
# this works on a list,
def Distance_JSD(S,group_act, distance='correlation'):
    '''ds_jsd'''
    NotImplementedError

def Distance(S,group_act, distance='correlation'):
    """ds"""
    if all([isinstance(x['activations'], xr.core.dataarray.DataArray) for x in group_act]):
        patterns_list = [x['activations'].transpose("presentation","neuroid_id")[dict(presentation=S)].values for x in group_act]
    else:
        # backward compatibility
        group_act_mod=[]
        for act_dict in group_act:
            act_=[x[0] if isinstance(act_dict['activations'][0], list) else x for x in act_dict['activations']]
            group_act_mod.append(act_)
        patterns_list=[]
        for grp_act in group_act_mod:
            patterns_list.append(np.stack([grp_act[i] for i in S]))
        #patterns_list = [np.stack([x['activations'][i] for i in S]) for x in group_act]
        #patterns_list = [np.stack([x[i] for i in S]) for x in group_act_mod]
    #[x.values for x in patterns if type]
    rdm2_vec = second_order_rdm(patterns_list, True, distance)
    return rdm2_vec.mean()

def minus_Distance(S,group_act, distance='correlation'):
    """2-ds"""
    if all([isinstance(x['activations'], xr.core.dataarray.DataArray) for x in group_act]):
        patterns_list = [x['activations'].transpose("presentation","neuroid_id")[dict(presentation=S)].values for x in group_act]
    else:
        # backward compatibility
        group_act_mod=[]
        for act_dict in group_act:
            act_=[x[0] if isinstance(act_dict['activations'][0], list) else x for x in act_dict['activations']]
            group_act_mod.append(act_)
        patterns_list=[]
        for grp_act in group_act_mod:
            patterns_list.append(np.stack([grp_act[i] for i in S]))
        #patterns_list = [np.stack([x['activations'][i] for i in S]) for x in group_act]
        #patterns_list = [np.stack([x[i] for i in S]) for x in group_act_mod]
    #[x.values for x in patterns if type]
    rdm2_vec = second_order_rdm(patterns_list, True, distance)
    return 2-rdm2_vec.mean()


def compute_rdm(S,group_act,vector=True, distance='correlation'):
    if all([isinstance(x['activations'], xr.core.dataarray.DataArray) for x in group_act]):
        patterns_list = [x['activations'].transpose("presentation","neuroid_id")[dict(presentation=S)].values for x in group_act]
    else:
        # backward compatibility
        group_act_mod=[]
        for act_dict in group_act:
            act_=[x[0] if isinstance(act_dict['activations'][0], list) else x for x in act_dict['activations']]
            group_act_mod.append(act_)
        patterns_list=[]
        for grp_act in tqdm(group_act_mod):
            patterns_list.append(np.stack([grp_act[i] for i in S]))
        #patterns_list = [np.stack([x['activations'][i] for i in S]) for x in group_act]
        #patterns_list = [np.stack([x[i] for i in S]) for x in group_act_mod]
    #[x.values for x in patterns if type]
    rdm1_vec=[]
    for  pattern_ in tqdm(patterns_list):
        rdm1_vec.append(create_rdm(pattern_, vec=vector, distance=distance))

    rdm2_vec = rdm2_vec = second_order_rdm(patterns_list, vec=vector, distance=distance)
    rdm_dict= dict(RDM_1st=rdm1_vec,RDM_2nd=rdm2_vec)
    return rdm_dict

@torch.no_grad()
def pt_create_corr_rdm_short(X,Y=None,vec=False,device=None):
    # note currently it is compeletely ignoring Y
    X=(X-X.mean(axis=1,keepdim=True))
    X=torch.nn.functional.normalize(X)
    if Y is not None:
            Y=(Y-Y.mean(axis=1,keepdim=True))
            Y=torch.nn.functional.normalize(Y)
    else:
        Y=X
    XY_corr=torch.tensor(1,device=X.device,dtype = float,requires_grad=False)-torch.mm(X,torch.transpose(Y,1,0))
    XY_corr=torch.triu(XY_corr,diagonal=1)
    if vec:
        return torch.clamp(torch.reshape(XY_corr,(1,-1)), 0.0, np.inf)
    return torch.clamp(XY_corr, 0.0, np.inf)
def Variation(s,N_S, pZ_S):
    qS = np.zeros(N_S)
    qS[s] = 1 / len(s)
    qZ = pZ_S.T @ qS
    return entropy(qZ)

# functionality for computing jenson-shannon divergence
def js_divergence(x_ref, x,bins=50,epsilon=1e-10):
    """
    Compute Jensen-Shannon Divergence (JSD) between two sets of samples.
    """
    # Compute histograms
    hist_ref = torch.histc(x_ref, bins=bins, min=0, max=2)
    hist_x = torch.histc(x, bins=bins, min=0, max=2)
    # Compute probabilities
    p = hist_ref / torch.sum(hist_ref)
    q = hist_x / torch.sum(hist_x)
    # Compute average probabilities
    p_smooth = p + epsilon
    q_smooth = q + epsilon
    # Normalize probabilities
    p_smooth /= p_smooth.sum()
    q_smooth /= q_smooth.sum()
    # Compute the average distribution
    m = 0.5 * (p_smooth + q_smooth)
    # Compute KL divergence between p and m
    kl_div_pm = F.kl_div(p_smooth.log(), m, reduction='batchmean')
    # Compute KL divergence between q and m
    kl_div_qm = F.kl_div(q_smooth.log(), m, reduction='batchmean')
    # Compute JSD
    jsd = 0.5 * (kl_div_pm + kl_div_qm)
    return jsd


class optim:
    def __init__(self, n_init=3, n_iter=300,N_s=50, objective_function=None,
                 rdm_function=compute_rdm, optim_algorithm=None,low_dim=False,pca_type='pytorch',pca_var=0.9,
                 run_gpu=False,early_stopping=True,stop_threshold=1e-4,jds_threshold=0.1,jsd_muliplier=0.5,device=None):
        self.n_iter=n_iter
        self.n_init=n_init
        self.N_s=N_s
        self.objective_function=objective_function
        self.rdm_function = compute_rdm
        self.optim_algorithm=optim_algorithm
        self.device=device
        self.run_gpu=run_gpu
        self.low_dim=low_dim
        self.pca_type=pca_type
        self.pca_var=pca_var
        self.early_stopping=early_stopping
        self.stop_threshold=stop_threshold
        self.jsd_threshold=jds_threshold
        self.jsd_muliplier=jsd_muliplier

    def load_extractor(self,extractor_obj=None):
        self.extractor_obj=extractor_obj
        self.N_S=extractor_obj.N_S
        self.extract_type = extractor_obj.extract_type
        self.activations = extractor_obj.model_group_act
        if self.extract_type == 'brain_resp':
            self.construct_activation_by_split()


    def construct_activation_by_split(self):
        #assert(len(self.activations[0])>1)
        model_grp_activation_by_split=[]
        for k in range(len(self.activations[0])): # going over the splits
            model_grp_activations=[]
            for activation_set in self.activations: # going over the models
                activation = dict(model_name=activation_set[k]['model_name'], layer=activation_set[k]['layer'], activations=activation_set[k]['activations'])
                model_grp_activations.append(activation)
            model_grp_activation_by_split.append(model_grp_activations)
        self.activations_by_split=model_grp_activation_by_split
        pass
    # TODO : Do the regression on train on all the data.
    # TODO : verify that Model activation, and Brain response,

    def compute_activation_in_low_dim(self,low_dim_num=300,low_resolution=False):
        #assert (torch.cuda.is_available())
        activation_list = []
        var_explained = []
        pca_type = 'fixed'
        for idx, act_dict in (enumerate(self.activations)):
            # backward compatibility
            act_=[x[0] if isinstance(act_dict['activations'][0], list) else x for x in act_dict['activations']]
            act = torch.tensor(act_, dtype=float, device=self.device, requires_grad=False)
            # act must be in m sample * n feature shape ,
            u, s, v = torch.pca_lowrank(act, q=500)
            # keep 85% variance explained ,
            idx_85 = torch.cumsum(s ** 2, dim=0) / torch.sum(s ** 2) < .85
            cols = list(torch.where(idx_85)[0].cpu().numpy())
            if pca_type == 'fixed':
                act_pca = torch.matmul(act, v[:, :low_dim_num])
            elif pca_type == 'equal_var':
                act_pca = torch.matmul(act, v[:, cols])

            activation_list.append(act_pca)
            var_explained.append(torch.cumsum(torch.cat((torch.tensor([0], device=self.device), s ** 2)),
                                              dim=0) / torch.sum(s ** 2))
        self.var_explained = var_explained
        if low_resolution == True:
            self.activation_list = [x.half() for x in activation_list]
        else:
            self.activation_list = activation_list

    def precompute_corr_rdm_on_gpu(self,low_resolution=False,cpu_dump=False,save_results=True,preload=True):
        #assert(torch.cuda.is_available())
        #torch.cuda.empty_cache()
        self.XY_corr_list=[]
        xy_dir = os.path.join(SAVE_DIR,
                              f"{self.extractor_obj.identifier}_XY_corr_list-low_res={low_resolution}-low_dim={self.low_dim}-pca_type={self.pca_type}-pca_var={self.pca_var}.pkl")
        if not cpu_dump:
            target_device = self.device
        else:
            target_device = torch.device('cpu')
        if preload:

            if os.path.exists(xy_dir):
                self.XY_corr_list=load_obj(xy_dir)
                self.XY_corr_list=[x.to(target_device) for x in self.XY_corr_list]
            else:
                assert False, "file doesnt exist, set preload to False"
        else:
            if self.low_dim:
                var_explained = []
                for idx, act_dict in tqdm(enumerate(self.activations)):
                    # backward compatibility
                    act_ = [x[0] if isinstance(act_dict['activations'][0], list) else x for x in act_dict['activations']]
                    act = torch.tensor(act_, dtype=float, device=self.device,requires_grad=False)
                    act_pca,var_exp=low_dim_project(act,var_explained=self.pca_var,pca_type=self.pca_type)
                    var_explained.append(var_exp)
                    XY_corr=corrcoef_metric(act_pca)
                    self.XY_corr_list.append(XY_corr.to(target_device))
                    del act
                    del act_pca
                    del XY_corr
                    torch.cuda.empty_cache()
                self.var_explained=var_explained

            else:
                for idx, act_dict in tqdm(enumerate(self.activations)):
                    # backward compatiblity
                    act_ = [x[0] if isinstance(act_dict['activations'][0], list) else x for x in act_dict['activations']]
                    act = torch.tensor(act_, dtype=float, device=self.device,requires_grad=False)
                    XY_corr=corrcoef_metric(act)
                    self.XY_corr_list.append(XY_corr.to(target_device))
                    del act
                    del XY_corr
                    torch.cuda.empty_cache()

        # double check target device allocation.
        self.XY_corr_list=[x.to(target_device) for x in self.XY_corr_list]
        if self.run_gpu:
            # delete activations from gpu if it exists in self
            self.activations = None
            torch.cuda.empty_cache()
        if save_results:
            D_precompute=self.XY_corr_list
            save_obj(D_precompute, xy_dir)

    def gpu_object_function_ds(self,S):
        samples=torch.tensor(S, dtype = torch.long, device = self.device)
        pairs = torch.combinations(samples, with_replacement=False)
        XY_corr_sample = [XY_corr[pairs[:, 0], pairs[:, 1]] for XY_corr in self.XY_corr_list]
        XY_corr_sample_tensor = torch.stack(XY_corr_sample).to(self.device)
        XY_corr_sample_tensor = torch.transpose(XY_corr_sample_tensor, 1, 0)
        if XY_corr_sample_tensor.shape[1] < XY_corr_sample_tensor.shape[0]:
            XY_corr_sample_tensor = torch.transpose(XY_corr_sample_tensor, 1, 0)
        assert (XY_corr_sample_tensor.shape[1] > XY_corr_sample_tensor.shape[0])
        d_mat = pt_create_corr_rdm_short(XY_corr_sample_tensor, device=self.device)
        n1 = d_mat.shape[1]
        correction = n1 * n1 / (n1 * (n1 - 1) / 2)
        d_val = correction * d_mat.mean(dim=(0, 1))
        d_val_mean=d_val.cpu().numpy().mean()
        # do a version with std reductions too
        mdl_pairs = torch.combinations(torch.tensor(np.arange(d_mat.shape[0])), with_replacement=False)
        d_val_std=torch.std(d_mat[mdl_pairs[:,0],mdl_pairs[:,1]]).cpu().numpy()
        d_optim=d_val_mean #-.2*d_val_std
        return d_optim

    def gpu_object_function_ds_plus_jsd(self,S,debug=False):
        samples=torch.tensor(S, dtype = torch.long, device = self.device)
        # use torch to select the samples
        samples_rand = torch.randperm(self.N_S )[:self.N_s]

        pairs = torch.combinations(samples, with_replacement=False)
        pairs_rand = torch.combinations(samples_rand, with_replacement=False)
        XY_corr_sample = [XY_corr[pairs[:, 0], pairs[:, 1]] for XY_corr in self.XY_corr_list]
        XY_corr_sample_tensor = torch.stack(XY_corr_sample).to(self.device)
        XY_corr_sample_tensor = torch.transpose(XY_corr_sample_tensor, 1, 0)
        if XY_corr_sample_tensor.shape[1] < XY_corr_sample_tensor.shape[0]:
            XY_corr_sample_tensor = torch.transpose(XY_corr_sample_tensor, 1, 0)
        assert (XY_corr_sample_tensor.shape[1] > XY_corr_sample_tensor.shape[0])
        # do the same for pairs rand
        XY_corr_sample_rand = [XY_corr[pairs_rand[:, 0], pairs_rand[:, 1]] for XY_corr in self.XY_corr_list]
        XY_corr_sample_tensor_rand = torch.stack(XY_corr_sample_rand).to(self.device)
        XY_corr_sample_tensor_rand = torch.transpose(XY_corr_sample_tensor_rand, 1, 0)
        if XY_corr_sample_tensor_rand.shape[1] < XY_corr_sample_tensor_rand.shape[0]:
            XY_corr_sample_tensor_rand = torch.transpose(XY_corr_sample_tensor_rand, 1, 0)
        assert (XY_corr_sample_tensor_rand.shape[1] > XY_corr_sample_tensor_rand.shape[0])

        # compute d_s for samples
        d_mat = pt_create_corr_rdm_short(XY_corr_sample_tensor, device=self.device)
        n1 = d_mat.shape[1]
        correction = n1 * n1 / (n1 * (n1 - 1) / 2)
        d_val = correction * d_mat.mean(dim=(0, 1))
        d_val_mean=d_val.cpu().numpy().mean()
        # do a version with std reductions too
        mdl_pairs = torch.combinations(torch.tensor(np.arange(d_mat.shape[0])), with_replacement=False)
        d_val_std=torch.std(d_mat[mdl_pairs[:,0],mdl_pairs[:,1]]).cpu().numpy()
        d_optim=d_val_mean #-.2*d_val_std

        # compute jsd for samples
        jsd_vals=[]
        for x, y in zip(XY_corr_sample_tensor, XY_corr_sample_tensor_rand):
            jsd_val = js_divergence(x, y)
            jsd_vals.append(jsd_val)
        jsd_ = torch.stack(jsd_vals).mean().cpu().numpy().mean()
        if jsd_<self.jsd_threshold:
            jsd_=0.0
        else:
            jsd_=-self.jsd_muliplier*jsd_

        if debug:
            return d_optim,jsd_,jsd_vals
        else:
            return d_optim+jsd_

    def gpu_object_function_minus_ds(self,S):
        samples=torch.tensor(S, dtype = torch.long, device = self.device)
        pairs = torch.combinations(samples, with_replacement=False)
        XY_corr_sample = [XY_corr[pairs[:, 0], pairs[:, 1]] for XY_corr in self.XY_corr_list]
        XY_corr_sample_tensor = torch.stack(XY_corr_sample).to(device)
        XY_corr_sample_tensor = torch.transpose(XY_corr_sample_tensor, 1, 0)
        if XY_corr_sample_tensor.shape[1] < XY_corr_sample_tensor.shape[0]:
            XY_corr_sample_tensor = torch.transpose(XY_corr_sample_tensor, 1, 0)
        assert (XY_corr_sample_tensor.shape[1] > XY_corr_sample_tensor.shape[0])
        d_mat = pt_create_corr_rdm_short(XY_corr_sample_tensor, device=self.device)
        n1 = d_mat.shape[1]
        correction = n1 * n1 / (n1 * (n1 - 1) / 2)
        d_val = correction * d_mat.mean(dim=(0, 1))
        d_val_mean=d_val.cpu().numpy().mean()
        # do a version with std reductions too
        mdl_pairs = torch.combinations(torch.tensor(np.arange(d_mat.shape[0])), with_replacement=False)
        d_val_std=torch.std(d_mat[mdl_pairs[:,0],mdl_pairs[:,1]]).cpu().numpy()
        d_optim=2-d_val_mean #-.2*d_val_std
        return d_optim

    def gpu_object_function_debug(self,S):
        samples = torch.tensor(S, dtype=torch.long, device=self.device)
        pairs = torch.combinations(samples, with_replacement=False)
        XY_corr_sample = [XY_corr[pairs[:, 0], pairs[:, 1]] for XY_corr in self.XY_corr_list]
        XY_corr_sample_tensor = torch.stack(XY_corr_sample).to(device)
        XY_corr_sample_tensor = torch.transpose(XY_corr_sample_tensor, 1, 0)
        if XY_corr_sample_tensor.shape[1] < XY_corr_sample_tensor.shape[0]:
            XY_corr_sample_tensor = torch.transpose(XY_corr_sample_tensor, 1, 0)
        assert (XY_corr_sample_tensor.shape[1] > XY_corr_sample_tensor.shape[0])
        d_mat = pt_create_corr_rdm_short(XY_corr_sample_tensor, device=self.device)
        n1 = d_mat.shape[1]
        correction = n1 * n1 / (n1 * (n1 - 1) / 2)
        d_val = correction * d_mat.mean(dim=(0, 1))
        d_val_mean = d_val.cpu().numpy().mean()

        return d_val_mean,d_mat

    def mod_objective_function(self,S):
        if self.extract_type=='activation':
            return self.objective_function(S,self.activations)
        elif self.extract_type=='brain_resp':
            return np.mean([self.objective_function(S,x) for x in self.activations_by_split])

    def mod_rdm_function(self,S,vector=True):
        if self.extract_type=='activation':
            return self.rdm_function(S,self.activations,vector=vector)
        elif self.extract_type=='brain_resp':
            return np.mean([self.rdm_function(S,x,vector=vector) for x in self.activations_by_split])

    def __call__(self,*args, **kwargs):

        if self.run_gpu:
            if self.objective_function.__doc__ == 'ds':
                objective = self.gpu_object_function_ds
            elif self.objective_function.__doc__ == '2-ds':
                objective = self.gpu_object_function_minus_ds
            elif self.objective_function.__doc__ == 'ds_jsd':
                objective = self.gpu_object_function_ds_plus_jsd

            if self.early_stopping:
                S_opt_d, DS_opt_d = self.optim_algorithm(N=self.N_S, n=self.N_s, objective_function=objective, n_init=self.n_init,
                                                         n_iter=self.n_iter,early_stopping=self.early_stopping,stop_threshold=self.stop_threshold)
            else:
                S_opt_d, DS_opt_d = self.optim_algorithm(self.N_S, self.N_s, objective, self.n_init,
                                                     self.n_iter)
        else:
            if self.early_stopping:

                S_opt_d, DS_opt_d = self.optim_algorithm(N=self.N_S, n=self.N_s,
                                                             objective_function=self.mod_objective_function,
                                                             n_init=self.n_init,
                                                             n_iter=self.n_iter, early_stopping=self.early_stopping,
                                                             stop_threshold=self.stop_threshold)
            else:
                S_opt_d, DS_opt_d = self.optim_algorithm(N=self.N_S, n=self.N_s,
                                                         objective_function=self.mod_objective_function,
                                                         n_init=self.n_init, n_iter=self.n_iter)
        self.S_opt_d=S_opt_d
        self.DS_opt_d=DS_opt_d

        return S_opt_d, DS_opt_d

class optim_group:
    def __init__(self,n_init=3,extract_group_name=None,ext_group_ids=[], n_iter=300,N_s=50, objective_function=Distance, optim_algorithm=None,run_gpu=False):
        self.n_iter = n_iter
        self.n_init = n_init
        self.N_s = N_s
        self.objective_function = objective_function
        self.optim_algorithm = optim_algorithm
        self.device = device
        self.run_gpu = run_gpu
        self.ext_group_ids=ext_group_ids
        self.extract_group_name=extract_group_name
        self.optim_obj=optim(optim_algorithm=self.optim_algorithm,objective_function=self.objective_function,n_init=self.n_init,n_iter=self.n_iter,run_gpu=self.run_gpu,N_s=self.N_s)

    def load_extr_grp_and_corr_rdm_in_low_dim(self,low_dim_num=200,low_resolution=True,cpu_dump=True,save_results=True):
        self.grp_XY_corr_list=[]
        for id_,ext_id in tqdm(enumerate(self.ext_group_ids)):
            # load extractor
            ext_obj=extract_pool[ext_id]()
            ext_obj.load_dataset()
            ext_obj()
            # load optim
            self.optim_obj = optim(optim_algorithm=self.optim_algorithm, objective_function=self.objective_function,
                                   n_init=self.n_init, n_iter=self.n_iter, run_gpu=self.run_gpu, N_s=self.N_s)
            self.optim_obj.load_extractor(ext_obj)
            del ext_obj
            self.N_S=self.optim_obj.N_S
            self.optim_obj.precompute_corr_rdm_on_gpu(low_dim=low_dim_num,low_resolution=low_resolution,cpu_dump=cpu_dump)
            self.grp_XY_corr_list.append(torch.stack(self.optim_obj.XY_corr_list))
            del self.optim_obj
        if save_results:
            D_precompute=dict(N_S=self.N_S,grp_XY_corr_list=self.grp_XY_corr_list)
            save_obj(D_precompute, os.path.join(SAVE_DIR, f"{self.extract_group_name}_XY_corr_list.pkl"))

        pass


    def XY_corr_obj_func(self,S,XY_corr_list,gpu_dump=False):
        samples = torch.tensor(S, dtype=torch.long, device=self.device)
        pairs = torch.combinations(samples, with_replacement=False)
        XY_corr_sample = [XY_corr[pairs[:, 0], pairs[:, 1]] for XY_corr in XY_corr_list]
        XY_corr_sample_tensor = torch.stack(XY_corr_sample).float()
        #XY_corr_sample_tensor = torch.transpose(XY_corr_sample_tensor, 1, 0)
        #if XY_corr_sample_tensor.shape[1] < XY_corr_sample_tensor.shape[0]:
        #    XY_corr_sample_tensor = torch.transpose(XY_corr_sample_tensor, 1, 0)
        assert (XY_corr_sample_tensor.shape[1] > XY_corr_sample_tensor.shape[0])
        d_mat = pt_create_corr_rdm_short(XY_corr_sample_tensor, device=self.device)
        n1 = d_mat.shape[1]
        correction = n1 * n1 / (n1 * (n1 - 1) / 2)
        d_val = correction * d_mat.mean(dim=(0, 1))
        d_val_mean = d_val.cpu().numpy().mean()
        # do a version with std reductions too
        mdl_pairs = torch.combinations(torch.tensor(np.arange(d_mat.shape[0])), with_replacement=False)
        d_val_std = torch.std(d_mat[mdl_pairs[:, 0], mdl_pairs[:, 1]]).cpu().numpy()
        d_optim = d_val_mean # - .2 * d_val_std
        del XY_corr_list
        return d_optim

    def gpu_obj_function(self,S):
        self.d_optim_list=[]
        for XY_corr_list in self.grp_XY_corr_list:
            self.d_optim_list.append(self.XY_corr_obj_func(S,XY_corr_list=XY_corr_list))
        self.d_optim=np.mean(self.d_optim_list)
        return self.d_optim



    def __call__(self, *args, **kwargs):

        if self.run_gpu:
            # push the data into gpu device
            self.grp_XY_corr_list=[x.to(device) for x in self.grp_XY_corr_list]
            [f"{x.device}" for x in self.grp_XY_corr_list]
            S_opt_d, DS_opt_d = self.optim_algorithm(self.N_S, self.N_s, self.gpu_obj_function, self.n_init,
                                                     self.n_iter)
        else :
            print(f"class is not defined for non gpu operations")
            S_opt_d=[]
            DS_opt_d=0

        self.S_opt_d=S_opt_d
        self.DS_opt_d=DS_opt_d
        return S_opt_d, DS_opt_d

optim_method=[dict(name='coordinate_ascent',fun=coordinate_ascent),
              dict(name='coordinate_ascent_eh',fun=coordinate_ascent_eh),
              dict(name='coordinate_ascent_parallel_eh',fun=coordinate_ascent_parallel_eh)]

objective_function=[dict(name='D_s',fun=Distance),dict(name='D_s_var',fun=Distance),
                    dict(name='2-D_s',fun=minus_Distance),dict(name='D_s_jsd',fun=Distance_JSD)]

n_iters=[2,50,100,500]
N_s=[10,25,50,75,80,100,125,150,175,200,225,250,275,300]
n_inits=[1,2]
run_gpu=[True,False]
low_dim=[True,False]
pca_var=[.9,.95]
pca_type=['pytorch','sklearn']

optim_configuration=[]
for method , obj,n_iter, n_s, init ,dim_, gpu,var_,type_ in itertools.product(optim_method,objective_function,n_iters,N_s,n_inits,low_dim,run_gpu,pca_var,pca_type):
    identifier=f"[{method['name']}]-[obj={obj['name']}]-[n_iter={n_iter}]-[n_samples={n_s}]-[n_init={init}]-[low_dim={dim_}]-[pca_var={var_}]-[pca_type={type_}]-[run_gpu={gpu}]"
    identifier=identifier.translate(str.maketrans({'[': '', ']': '', '/': '_'}))
    optim_configuration.append(dict(identifier=identifier,method=method['fun'],obj=obj['fun'],n_iter=n_iter,n_s=n_s,n_init=init,low_dim=dim_,run_gpu=gpu,pca_var=var_,pca_type=type_))


optim_pool={}
# create the pool
for config in optim_configuration:
    configuration=copy.deepcopy(config)
    optim_identifier=configuration['identifier']
    def optim_instantiation(configure=frozenset(configuration.items())):
        configure = dict(configure)
        #module = import_module('utils.model_utils')
        #model=getattr(module,configure['model'])
        optim_param=optim(optim_algorithm=configure['method'],
                          objective_function=configure['obj'],
                                  n_init=configure['n_init'],
                                  n_iter=configure['n_iter'],
                                run_gpu=configure['run_gpu'],
                                low_dim=configure['low_dim'],
                                pca_var=configure['pca_var'],
                                pca_type=configure['pca_type'],
                                  N_s=configure['n_s'])
        return optim_param

    optim_pool[optim_identifier] = optim_instantiation





