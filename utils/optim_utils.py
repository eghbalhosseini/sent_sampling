import sys
import numpy as np
from scipy.stats import entropy
import logging
import getpass
import itertools
import copy
import xarray as xr
import torch
from tqdm import tqdm
import utils
from utils import extract_pool
device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
try :
    torch.set_deterministic(True)
except:
    pass
torch.set_printoptions(precision=10)
from scipy.spatial.distance import squareform
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if getpass.getuser()=='eghbalhosseini':
    OPTIM_PARENT='/Users/eghbalhosseini/MyCodes/opt-exp-design-nlp/'
elif getpass.getuser()=='ehoseini':
    OPTIM_PARENT = '/om/user/ehoseini/opt-exp-design-nlp'
elif getpass.getuser()=='alexso':
    OPTIM_PARENT = '/om/user/alexso/opt-exp-design-nlp'




LOG_BASE = np.e
EPSILON = 1e-16
PRECISION = 1e-200

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s - %(funcName)s] %(message)s')
ch.setFormatter(formatter)

#‌impor functions from Noga's optimization pipeline.
sys.path.insert(1,OPTIM_PARENT)
from opt_exp_design import coordinate_ascent, coordinate_ascent_eh,coordinate_ascent_parallel_eh
import tools
from tools import second_order_rdm, create_rdm, MI
LOGGER = tools.get_logger('OPT-EXP-DSGN')

# Objective functions :
# this works on a list,
def Distance(S,group_act, distance='correlation'):
    if all([isinstance(x['activations'], xr.core.dataarray.DataArray) for x in group_act]):
        patterns_list = [x['activations'].transpose("presentation","neuroid_id")[dict(presentation=S)].values for x in group_act]
    else:
        patterns_list = [np.stack([x['activations'][i] for i in S]) for x in group_act]
    #[x.values for x in patterns if type]
    rdm2_vec = second_order_rdm(patterns_list, True, distance)
    return rdm2_vec.mean()

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
    XY_corr=torch.tensor(1,device=device,dtype = float,requires_grad=False)-torch.mm(X,torch.transpose(Y,1,0))
    XY_corr=torch.triu(XY_corr,diagonal=1)
    if vec:
        return torch.clamp(torch.reshape(XY_corr,(1,-1)), 0.0, np.inf)
    return torch.clamp(XY_corr, 0.0, np.inf)

def Variation(s,N_S, pZ_S):
    qS = np.zeros(N_S)
    qS[s] = 1 / len(s)
    qZ = pZ_S.T @ qS
    return entropy(qZ)




class optim:
    def __init__(self, n_init=3, n_iter=300,N_s=50, objective_function=Distance, optim_algorithm=None,run_gpu=False):
        self.n_iter=n_iter
        self.n_init=n_init
        self.N_s=N_s
        self.objective_function=objective_function
        self.optim_algorithm=optim_algorithm
        self.device=device
        self.run_gpu=run_gpu

    def load_extractor(self,extractor_obj=None):
        self.extractor_obj=extractor_obj
        self.N_S=extractor_obj.N_S
        self.extract_type = extractor_obj.extract_type
        self.activations = extractor_obj.model_group_act
        if self.extract_type == 'brain_resp':
            self.construct_activation_by_split()


    # add a function for random baseline

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
        assert (torch.cuda.is_available())
        activation_list = []
        var_explained = []
        pca_type = 'fixed'
        for idx, act_dict in (enumerate(self.activations)):

            act = torch.tensor(act_dict['activations'], dtype=float, device=self.device, requires_grad=False)
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

    def precompute_corr_rdm_on_gpu(self,low_dim=False,low_dim_num=300,pca_type='fixed',low_resolution=False,cpu_dump=False):
        assert(torch.cuda.is_available())
        self.XY_corr_list=[]
        if not cpu_dump:
            target_device=self.device
        else:
            target_device=torch.device('cpu')
        if low_dim:
            var_explained = []
            for idx, act_dict in (enumerate(self.activations)):

                act = torch.tensor(act_dict['activations'], dtype=float, device=self.device,requires_grad=False)
                # act must be in m sample * n feature shape ,
                u, s, v = torch.pca_lowrank(act, q=500)
                # keep 85% variance explained ,
                idx_85 = torch.cumsum(s ** 2, dim=0) / torch.sum(s ** 2) < .85
                cols = list(torch.where(idx_85)[0].cpu().numpy())
                if pca_type == 'fixed':
                    act_pca = torch.matmul(act, v[:, :low_dim_num])
                elif pca_type == 'equal_var':
                    act_pca = torch.matmul(act, v[:, cols])

                #activation_list.append(act_pca)
                var_explained.append(torch.cumsum(torch.cat((torch.tensor([0], device=self.device), s ** 2)),
                                                  dim=0) / torch.sum(s ** 2))
                # just in time computation:
                X=torch.nn.functional.normalize(act_pca.squeeze())
                X=X - X.mean(axis=1, keepdim=True)
                X =torch.nn.functional.normalize(X)
                if low_resolution==True:
                    XY_corr = torch.tensor(1, device=self.device, dtype=torch.float16) - torch.mm(X, torch.transpose(X, 1, 0)).half()
                else:
                    XY_corr = torch.tensor(1, device=self.device, dtype=float) - torch.mm(X,torch.transpose(X, 1,0))

                self.XY_corr_list.append(XY_corr)
            self.var_explained=var_explained
        else:
            for idx, act_dict in (enumerate(self.activations)):
                act = torch.tensor(act_dict['activations'], dtype=float, device=self.device,requires_grad=False)
                X = torch.nn.functional.normalize(act.squeeze())
                X = X - X.mean(axis=1, keepdim=True)
                X = torch.nn.functional.normalize(X)
                if low_resolution == True:
                    XY_corr = torch.tensor(1, device=self.device, dtype=torch.float16) - torch.mm(X,torch.transpose(X, 1,0)).half()
                else:
                    XY_corr = torch.tensor(1, device=self.device, dtype=float) - torch.mm(X, torch.transpose(X, 1, 0))

                self.XY_corr_list.append(XY_corr)
        self.XY_corr_list=[x.to(target_device) for x in self.XY_corr_list]
        if self.run_gpu:
            del self.activations
        pass

    def gpu_object_function(self,S):
        samples=torch.tensor(S, dtype = torch.long, device = self.device)
        pairs = torch.combinations(samples, with_replacement=False)
        XY_corr_sample = [XY_corr[pairs[:, 0], pairs[:, 1]] for XY_corr in self.XY_corr_list]
        XY_corr_sample_tensor = torch.stack(XY_corr_sample)
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

    def gpu_object_function_debug(self,S):
        samples=torch.tensor(S, dtype = torch.long, device = self.device)
        pairs = torch.combinations(samples, with_replacement=False)
        XY_corr_sample = [XY_corr[pairs[:, 0], pairs[:, 1]] for XY_corr in self.XY_corr_list]
        XY_corr_sample_tensor = torch.stack(XY_corr_sample)
        XY_corr_sample_tensor = torch.transpose(XY_corr_sample_tensor, 1, 0)

        XY_sample = [torch.index_select(XY_corr, 0, samples) for XY_corr in self.XY_corr_list]
        XY_sample = [torch.index_select(XY_s, 1, samples) for XY_s in XY_sample]

        if XY_corr_sample_tensor.shape[1] < XY_corr_sample_tensor.shape[0]:
            XY_corr_sample_tensor = torch.transpose(XY_corr_sample_tensor, 1, 0)
        assert (XY_corr_sample_tensor.shape[1] > XY_corr_sample_tensor.shape[0])
        d_mat = pt_create_corr_rdm_short(XY_corr_sample_tensor, device=self.device)
        n1 = d_mat.shape[1]
        correction = n1 * n1 / (n1 * (n1 - 1) / 2)
        d_val = correction * d_mat.mean(dim=(0, 1))
        return d_val.cpu().numpy().mean(),XY_corr_sample_tensor, XY_sample

    def mod_objective_function(self,S):
        if self.extract_type=='activation':
            return self.objective_function(S,self.activations)
        elif self.extract_type=='brain_resp':
            return np.mean([self.objective_function(S,x) for x in self.activations_by_split])

    def __call__(self,*args, **kwargs):
        if self.run_gpu:
            S_opt_d, DS_opt_d = self.optim_algorithm(self.N_S, self.N_s, self.gpu_object_function, self.n_init,
                                                     self.n_iter)
        else:
            S_opt_d, DS_opt_d = self.optim_algorithm(self.N_S, self.N_s,self.mod_objective_function, self.n_init, self.n_iter)
        self.S_opt_d=S_opt_d
        self.DS_opt_d=DS_opt_d

        return S_opt_d, DS_opt_d



optim_method=[dict(name='coordinate_ascent',fun=coordinate_ascent),
              dict(name='coordinate_ascent_eh',fun=coordinate_ascent_eh),
              dict(name='coordinate_ascent_parallel_eh',fun=coordinate_ascent_parallel_eh)]

objective_function=[dict(name='D_s',fun=Distance),dict(name='D_s_var',fun=Distance)]

n_iters=[2,5,25,50,100,500,1000,2000,5000,10000]
N_s=[10,25,50,75,100,125,150,175,200,225,250,275,300]
n_inits=[1,2,3,5]
run_gpu=[True,False]


optim_configuration=[]
for method , obj,n_iter, n_s, init , gpu in itertools.product(optim_method,objective_function,n_iters,N_s,n_inits,run_gpu):
    identifier=f"[{method['name']}]-[obj={obj['name']}]-[n_iter={n_iter}]-[n_samples={n_s}]-[n_init={init}]-[run_gpu={gpu}]"
    identifier=identifier.translate(str.maketrans({'[': '', ']': '', '/': '_'}))
    optim_configuration.append(dict(identifier=identifier,method=method['fun'],obj=obj['fun'],n_iter=n_iter,n_s=n_s,n_init=init,run_gpu=gpu))


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
                                  N_s=configure['n_s'])
        return optim_param

    optim_pool[optim_identifier] = optim_instantiation


class optim_group:
    def __init__(self,n_init=3,ext_group_ids=[], n_iter=300,N_s=50, objective_function=Distance, optim_algorithm=None,run_gpu=False):
        self.n_iter = n_iter
        self.n_init = n_init
        self.N_s = N_s
        self.objective_function = objective_function
        self.optim_algorithm = optim_algorithm
        self.device = device
        self.run_gpu = run_gpu
        self.ext_group_ids=ext_group_ids
        self.optim_obj=optim(optim_algorithm=self.optim_algorithm,objective_function=self.objective_function,n_init=self.n_init,n_iter=self.n_iter,run_gpu=self.run_gpu,N_s=self.N_s)

    def load_extr_grp_and_corr_rdm_in_low_dim(self,low_dim_num=200,low_resolution=True):
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
            self.N_S=self.optim_obj.N_S
            self.optim_obj.precompute_corr_rdm_on_gpu(low_dim=low_dim_num,low_resolution=low_resolution,cpu_dump=True)
            self.grp_XY_corr_list.append(self.optim_obj.XY_corr_list)
            del self.optim_obj
        pass

    def XY_corr_obj_func(self,S,XY_corr_list):
        samples = torch.tensor(S, dtype=torch.long, device=self.device)
        pairs = torch.combinations(samples, with_replacement=False)
        XY_corr_sample = [XY_corr[pairs[:, 0], pairs[:, 1]] for XY_corr in XY_corr_list]
        XY_corr_sample_tensor = torch.stack(XY_corr_sample)
        XY_corr_sample_tensor = torch.transpose(XY_corr_sample_tensor, 1, 0)
        if XY_corr_sample_tensor.shape[1] < XY_corr_sample_tensor.shape[0]:
            XY_corr_sample_tensor = torch.transpose(XY_corr_sample_tensor, 1, 0)
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
        return d_optim

    def gpu_obj_function(self,S):
        self.d_optim_list=[]
        for XY_corr_list in self.grp_XY_corr_list:
            self.d_optim_list.append(self.XY_corr_obj_func(S,XY_corr_list=XY_corr_list))

        self.d_optim=np.mean(self.d_optim_list)
        return self.d_optim

    def __call__(self, *args, **kwargs):

        if self.run_gpu:
            S_opt_d, DS_opt_d = self.optim_algorithm(self.N_S, self.N_s, self.gpu_obj_function, self.n_init,
                                                     self.n_iter)
        else :
            print(f"class is not defined for non gpu operations")
            S_opt_d=[]
            DS_opt_d=0

        self.S_opt_d=S_opt_d
        self.DS_opt_d=DS_opt_d
        return S_opt_d, DS_opt_d






