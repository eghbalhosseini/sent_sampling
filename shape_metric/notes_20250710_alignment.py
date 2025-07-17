from datasets import load_dataset
#from platonic.
from collections import namedtuple
import sys
import os
# Add the root directory of the repo to sys.path
sys.path.append(os.path.abspath('/om2/user/ehoseini/platonic-rep'))
from extract_features import extract_llm_features, extract_lvm_features
from utils import to_feature_filename
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor


revision_d='wit_1024'
datas=load_dataset('minhuh/prh',revision=revision_d, split='train')


def mock_get_lvm_args():
    mock_args = namedtuple('debug', ['output_dir', 'pool','dataset','subset','force_remake','batch_size'])
    new_args = mock_args('/rdma/vast-rdma/vast/evlab/ehoseini/MyData/shape_metric', 'cls', revision_d, 'train', False, 4)
    return new_args

def mock_get_llm_args():
    mock_args = namedtuple('debug', ['output_dir', 'pool','dataset','subset','force_remake','batch_size','qlora','caption_idx','prompt',
                                     'force_download'])
    new_args = mock_args('/rdma/vast-rdma/vast/evlab/ehoseini/MyData/shape_metric', 'avg', revision_d, 'train', False, 4,True,0,False,True)
    return new_args



lvm_models = [
            "vit_tiny_patch16_224.augreg_in21k",
            "vit_small_patch16_224.augreg_in21k",
            "vit_base_patch16_224.augreg_in21k",
            "vit_large_patch16_224.augreg_in21k",
            "vit_base_patch16_224.mae",
            "vit_large_patch16_224.mae",
            "vit_huge_patch14_224.mae",
            "vit_small_patch14_dinov2.lvd142m",
            "vit_base_patch14_dinov2.lvd142m",
            "vit_large_patch14_dinov2.lvd142m",
            "vit_giant_patch14_dinov2.lvd142m",
            "vit_base_patch16_clip_224.laion2b",
            "vit_large_patch14_clip_224.laion2b",
            "vit_huge_patch14_clip_224.laion2b",
            "vit_base_patch16_clip_224.laion2b_ft_in12k",
            "vit_large_patch14_clip_224.laion2b_ft_in12k",
            "vit_huge_patch14_clip_224.laion2b_ft_in12k",
        ]

llm_models = [
    "openlm-research/open_llama_3b",
    "openlm-research/open_llama_7b",
    "openlm-research/open_llama_13b",
    "huggyllama/llama-7b",
    "huggyllama/llama-13b",

]

llm_models = [
    "bigscience/bloomz-560m",
    "bigscience/bloomz-1b1",
    "bigscience/bloomz-1b7",
    "bigscience/bloomz-3b",
    "bigscience/bloomz-7b1",
    "openlm-research/open_llama_3b",
    "openlm-research/open_llama_7b",
    "openlm-research/open_llama_13b",
    "huggyllama/llama-7b",
    "huggyllama/llama-13b",
    "huggyllama/llama-30b",
    "huggyllama/llama-65b",
]


for model_name in lvm_models:
    new_args = mock_get_lvm_args()
    model_names=[model_name,]
    extract_lvm_features(model_names, datas, new_args)


for model_name in llm_models:
    new_args = mock_get_llm_args()

    model_names=[model_name,]
    extract_llm_features(model_names, datas, new_args)


