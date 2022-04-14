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
    plt.hist(wave_tss, bins=100, edgecolor='k', linewidth=.2)
    plt.show()
    sent_length=[x['sentence_length'] for x in sent_info]
    #plt.scatter(wave_tss,sent_length)
    tss_ks=[]
    for k in np.unique(sent_length):
        locations=np.argwhere(np.asarray(sent_length)==k)
        tss_k=wave_tss[locations]
        tss_ks.append(np.squeeze(tss_k))
    ks=np.unique(sent_length)
    plt.violinplot(tss_ks,positions=ks)
    plt.show()
    #

    alowable_range=np.argwhere(np.logical_and(wave_tss<4.0, wave_tss>2.0))

    wave_tss_allowed=wave_tss[alowable_range]
    alow_sent_length=[sent_length[int(x)] for x in alowable_range]

    tss_ks_allow=[]
    for k in np.unique(alow_sent_length):
        locations=np.argwhere(np.asarray(alow_sent_length)==k)
        tss_k=wave_tss_allowed[locations]
        tss_ks_allow.append(np.squeeze(tss_k))
    ks=np.unique(alow_sent_length)
    plt.violinplot(tss_ks_allow,positions=ks)
    plt.show()


    plt.hist(np.squeeze(wave_tss[alowable_range]),bins=100,edgecolor='k', linewidth=.2)
    plt.show()
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