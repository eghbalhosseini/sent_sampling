import sys
sys.path.extend(['/om/user/ehoseini/sent_sampling', '/om/user/ehoseini/sent_sampling'])
from utils import extract_pool
from utils.optim_utils import optim_pool
import argparse
from utils.data_utils import RESULTS_DIR, save_obj,SAVE_DIR,load_obj, COCA_CORPUS_DIR
import torchaudio.transforms as T
import torchaudio.functional as F
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from glob import glob
import torchaudio
import librosa
from librosa.onset import onset_detect
import numpy as np
import re
import matplotlib as mpl
if __name__ == '__main__':
    audio_files_order=glob('/om/user/ehoseini/MyData/sent_sampling/coca_spok_filter_punct_50K_sylb/wav/coca_spok_filter_punct_50K_sylb_*.wav')
    len(audio_files_order)
    sent_id=[re.findall(r'sylb_\d+',x)[0] for x in audio_files_order]
    sentence_id=[int(x.replace('sylb_','')) for x in sent_id]
    reorder_=np.argsort(sentence_id)
    audio_files=[audio_files_order[x] for x in reorder_]
    sentence_id=np.sort(sentence_id)

    extractor_id = f'group=gpt2-xl_layers-dataset=coca_spok_filter_punct_50K_sylb-activation-bench=None-ave=False'
    extractor_obj = extract_pool[extractor_id]()
    extractor_obj.load_dataset()

    wave_tss=[]
    sent_info=[]

    for idx in tqdm(range(len(audio_files))):
        waveform,sample_rate=torchaudio.load(audio_files[idx])
        s_id=sentence_id[idx]
        sent_info.append(extractor_obj.data_[s_id])
        wave_ts=librosa.get_duration(filename=audio_files[idx])
        waveform,sample_rate= librosa.load(audio_files[idx])
        #wave_ts=waveform.shape[-1]/sample_rate
        wave_tss.append(wave_ts)
    wave_tss=np.asarray(wave_tss)

    sent_length = [x['sentence_length'] for x in sent_info]
    tss_ks=[]
    for k in np.unique(sent_length):
        locations=np.argwhere(np.asarray(sent_length)==k)
        tss_k=wave_tss[locations]
        tss_ks.append(np.squeeze(tss_k))
    ks=np.unique(sent_length)

    alowable_range = np.argwhere(np.logical_and(wave_tss < 4.0, wave_tss > 2.0))

    wave_tss_allowed = wave_tss[alowable_range]
    alow_sent_length = [sent_length[int(x)] for x in alowable_range]

    tss_ks_allow = []
    for k in np.unique(alow_sent_length):
        locations = np.argwhere(np.asarray(alow_sent_length) == k)
        tss_k = wave_tss_allowed[locations]
        tss_ks_allow.append(np.squeeze(tss_k))
    ks = np.unique(alow_sent_length)

    #
    #fig = plt.figure(figsize=(11, 8), dpi=100, frameon=False)
    fig = plt.figure(figsize=(11, 8), dpi=200, frameon=False)
    ax = plt.axes((.1, .5, .35, .25))
    ax.hist(wave_tss, bins=50, edgecolor='w', linewidth=.2,color=np.divide((55, 76, 128),256))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('count')
    ax.set_xlabel('duration (seconds)')
    ax.set_title('estimated duration of sentences')
    ax = plt.axes((.1, .1, .35, .25))
    viol=ax.violinplot(tss_ks,positions=ks,showmeans=True,showextrema=False)
    for pc in viol['bodies']:
        pc.set_facecolor(np.divide((55, 76, 128),256))
        pc.set_edgecolor('white')
        pc.set_alpha(1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('number of words')
    ax.set_ylabel('duration (seconds)')
    #ax.set_title('estimated duration of sentences')

    #
    ax = plt.axes((.55, .5, .35, .25))
    ax.hist(np.squeeze(wave_tss[alowable_range]), bins=50, edgecolor='w', linewidth=.2, color=np.divide((255, 166, 0), 255))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('count')
    ax.set_xlabel('duration (seconds)')
    ax.set_title('estimated duration of selected sentences')


    ax = plt.axes((.55, .1, .35, .25))
    viol=ax.violinplot(tss_ks_allow,positions=ks,showmeans=True,showextrema=False)
    for pc in viol['bodies']:
        pc.set_facecolor(np.divide((255, 166, 0), 255))
        pc.set_edgecolor('white')
        pc.set_alpha(1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('number of words')
    ax.set_ylabel('duration (seconds)')
    fig.show()
    # get sentence index for the alowable range
    allowable_sent_id=[sentence_id[int(x)] for x in alowable_range]
    allowable_sent_id[-1]
    extractor_obj.data_[19850]['text']

    for i, layer in tqdm(enumerate(range(49)), desc='layers'):
        model_activation_name = f"{extractor_obj.dataset}_{extractor_obj.model_spec[i]}_layer_{i}_{extractor_obj.extract_name}_ave_None.pkl"
        if os.path.exists(os.path.join(SAVE_DIR, model_activation_name)):
            t = load_obj(os.path.join(SAVE_DIR, model_activation_name))
            t_mod= [t[x] for x in allowable_sent_id]
            length_from_file=[x[0].shape[0] for x in t_mod]
            assert(np.array_equal(np.asarray(length_from_file),np.squeeze(alow_sent_length)))

            new_model_actvation_name = f"{extractor_obj.dataset}_sylb_2to4sec_{extractor_obj.model_spec[i]}_layer_{i}_{extractor_obj.extract_name}_ave_None.pkl"
            save_obj(t_mod, os.path.join(SAVE_DIR, new_model_actvation_name))
    #         assert(np.array_equal(pp,np.asarray([x[0].shape[0] for x in t_mod])))

            t_last=[]
            for id,x in tqdm(enumerate(t_mod),total=len(t_mod)):
                 mod=[x[0][[-1],:],x[1],x[2]]
                 t_last.append(mod)
            new_model_activation_name=f"{extractor_obj.dataset}_sylb_2to4sec_{extractor_obj.model_spec[i]}_layer_{i}_{extractor_obj.extract_name}_ave_False.pkl"
            save_obj(t_last, os.path.join(SAVE_DIR, new_model_activation_name))

    new_list=[extractor_obj.data_[x] for x in allowable_sent_id]


    new_dataset_id = f'coca_spok_data_filter_ngram_punct_50K_sylb_2to4sec.pkl'

    file_loc = os.path.join(COCA_CORPUS_DIR,new_dataset_id)
    save_obj(new_list,file_loc)

    '''read and save correaltion '''
    extractor_id = f'group=gpt2-xl_layers-dataset=coca_spok_filter_punct_50K_sylb_2to4sec-activation-bench=None-ave=False'
    extractor_obj = extract_pool[extractor_id]()
    extractor_obj.load_dataset()
    extractor_obj()
    optimizer_id = f"coordinate_ascent_eh-obj=D_s-n_iter=1000-n_samples=250-n_init=1-run_gpu=True"

    optimizer_obj = optim_pool[optimizer_id]()
    optimizer_obj.load_extractor(extractor_obj)

    optimizer_obj.precompute_corr_rdm_on_gpu(low_resolution=False, cpu_dump=True,preload=False)

    files_t_fix=glob(os.path.join(SAVE_DIR,'coca_spok_filter_punct_50K_sylb_sylb_2to4sec_gpt2-xl_layer_*.pkl'))
    for file in files_t_fix:
        os.rename(file, file.replace('_sylb_sylb_','_sylb_'))

    optimizer_obj = optim_pool[optimizer_id]()
    optimizer_obj.load_extractor(extractor_obj)
    S_opt_d, DS_opt_d = optimizer_obj()
    # plot_waveform(waveform, sample_rate)
    #
    # n_fft = 1024
    # win_length = None
    # hop_length = 512
    #
    # # define transformation
    # spectrogram = T.Spectrogram(
    #     n_fft=n_fft,
    #     win_length=win_length,
    #     hop_length=hop_length,
    #     center=True,
    #     pad_mode="reflect",
    #     power=2.0,
    # )
    # # Perform transformation
    # spec = spectrogram(waveform)
    #
    # a=librosa.feature.rms(np.squeeze(waveform))
    # plt.plot(np.squeeze(a))
    # plt.show()
    # plot_spectrogram(spec[0], title="torchaudio")