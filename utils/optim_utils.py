import sys
import numpy as np
from scipy.stats import entropy
import logging
import getpass
if getpass.getuser()=='eghbalhosseini':
    OPTIM_PARENT='/Users/eghbalhosseini/MyCodes/opt-exp-design-nlp/'
elif getpass.getuser()=='ehoseini':
    OPTIM_PARENT = '/om/user/ehoseini/opt-exp-design-nlp'




LOG_BASE = np.e
EPSILON = 1e-16
PRECISION = 1e-200

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s - %(funcName)s] %(message)s')
ch.setFormatter(formatter)

#‌impor functions from Noga's optimization pipeline.
sys.path.insert(1,OPTIM_PARENT)
from opt_exp_design import coordinate_ascent, swap
import tools
from tools import second_order_rdm, create_rdm, MI
LOGGER = tools.get_logger('OPT-EXP-DSGN')

# Objective functions :
# this works on a list,
def Distance(S,group_act, distance='correlation'):
    patterns_list = [[x['activations'][i] for i in S] for x in group_act]
    rdm2_vec = second_order_rdm(patterns_list, True, distance)
    return rdm2_vec.mean()


def Variation(s,N_S, pZ_S):
    qS = np.zeros(N_S)
    qS[s] = 1 / len(s)
    qZ = pZ_S.T @ qS
    return entropy(qZ)

def Mutual_Info_S(s,N_S, pZ_S):
    qS = np.zeros(N_S)
    qS[s] = 1 / len(s)
    qSZ = pZ_S * qS[:, None]
    return MI(qSZ)


class optim:
    def __init__(self, n_init=3, n_iter=300,N_s=50, objective_function=Distance, optim_algorthim=coordinate_ascent,extractor_obj=None):
        self.n_iter=n_iter
        self.n_init=n_init
        self.N_s=N_s
        self.objective_function=objective_function
        self.optim_algorithm=optim_algorthim
        self.extract_type=extractor_obj.extract_type
        self.activations = extractor_obj.model_group_act
        if self.extract_type == 'brain_resp':
            self.construct_activation_by_split()

    # add a function for random baseline

    def construct_activation_by_split(self):
        assert(len(self.activations[0])>1)
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
    def mod_objective_function(self,S):
        if self.extract_type=='network_act':
            return self.objective_function(S,self.activations)
        elif self.extract_type=='brain_resp':
            return np.mean([self.objective_function(S,x) for x in self.activations_by_split])

    def __call__(self,extractor_obj ,*args, **kwargs):

        S_opt_d, DS_opt_d = self.optim_algorithm(extractor_obj.N_S, self.N_s,self.mod_objective_function, self.n_init, self.n_iter)
        self.S_opt_d=S_opt_d
        self.DS_opt_d=DS_opt_d

        return S_opt_d, DS_opt_d





