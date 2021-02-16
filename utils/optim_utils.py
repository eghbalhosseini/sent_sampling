import sys
import numpy as np
from scipy.stats import entropy
import logging
import getpass
import itertools
import copy
import xarray as xr
import torch
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
    XY_corr=torch.tensor(1,device=device,dtype = float,requires_grad=False)-torch.mm(X,torch.transpose(X,1,0))
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

    def precompute_corr_rdm_on_gpu(self):
        assert(torch.cuda.is_available())
        activation_list = [torch.tensor(x['activations'], dtype=float, device=self.device, requires_grad=False) for x in
                           self.activations]
        X_list = [torch.nn.functional.normalize(x.squeeze()) for x in activation_list]
        X_list = [(X - X.mean(axis=1, keepdim=True)) for X in X_list]
        X_list = [torch.nn.functional.normalize(X) for X in X_list]
        self.XY_corr_list = [torch.tensor(1, device=self.device, dtype=float) - torch.mm(X, torch.transpose(X, 1, 0)) for X in
                        X_list]
        del X_list, activation_list
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
        return d_val.cpu().numpy().mean()

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
objective_function=[dict(name='D_s',fun=Distance)]

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





