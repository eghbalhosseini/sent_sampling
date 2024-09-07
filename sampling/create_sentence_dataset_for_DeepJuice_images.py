import numpy as np
from sent_sampling.utils import extract_pool
from sent_sampling.utils.extract_utils import model_extractor
from sent_sampling.utils.optim_utils import optim_pool
import argparse
from sent_sampling.utils.extract_utils import model_extractor, model_extractor_parallel
from sent_sampling.utils.data_utils import SENTENCE_CONFIG
from sent_sampling.utils.data_utils import RESULTS_DIR, save_obj, load_obj
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
import importlib
import sys

from sent_sampling.utils.data_utils import SENTENCE_CONFIG
import matplotlib
sys.path.append('/om2/user/ehoseini/DeepJuiceDev/')
deepjuice_path='/nese/mit/group/evlab/u/ehoseini/MyData/DeepJuice/'
from benchmarks import NSDBenchmark, NSDSampleBenchmark
from pathlib import Path
def check_no_space_words(sentences):
    for sentence in sentences:
        for word in sentence:
            if word.strip() == "":
                return False  # Found a word that is just spaces
    return True  # No words with only spaces found


if __name__ == '__main__':
    nsd_bench=NSDBenchmark()
    nsd_bench.stimulus_data.keys()
    captions_ALL = []
    for captions_set in nsd_bench.stimulus_data.coco_captions:
        True
        captions = captions_set.strip("[]")
        # split by
        captions= captions[1:-1]
        captions = captions.split("', '")
        # find and drop '"' character form elements in captions
        captions = [x.replace('\"', '') for x in captions]
        captions_ALL.append(captions)

    # create a random number generator for reproducibility
    # caption_selected=[]
    # for caption in captions_ALL:
    #     np.random.seed(0)
    #     np.random.shuffle(caption)
    #     # select the first caption from each set
    #     caption_selected.append(caption[0])

    caption_selected=[x[0] for x in captions_ALL]
    # make all the letters lowercase
    caption_selected = [x.lower() for x in caption_selected]
    # save caption_selected as a text file
    with open(deepjuice_path+'caption_selected.txt', 'w') as f:
        for item in caption_selected:
            f.write("%s\n" % item)
    # replace sentence :  'a small bus with the word "orbit" painted on it\\\'s side.', with 'a small bus with the word "orbit" painted on its side.'

    #caption_selected = [x.replace('it\\\'s', 'its') for x in caption_selected]

    # find the indx of stenose that is  'a yellow',
    indx = caption_selected.index('a white toilet sitting next ot a bathroom sink under a mirror.')
    caption_selected[indx]=captions_ALL[indx][2]
    indx=['a large clock tower with two golden clocks on ' in x for x in caption_selected].index(True)
    caption_selected[indx]=captions_ALL[indx][-1]
    indx = ['a head shot of a giraffe with another giraffe in the background' in x for x in caption_selected].index(True)
    caption_selected[indx]=captions_ALL[indx][-2]

    indx = ['players attempt to block a pass' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][-1]

    indx = ['a yellow bellied bird perched on a tree limb with its beak ope' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][-1]


    indx = ['a man riding on the back of a blac' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][-2]

    indx = ['two adults in a rec room playing with ' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][1].split(',')[0].replace('\'', '')

    indx = ['a young girl and suited man are in a school atmosphere' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][0].split(',')[0].replace('\'', '')

    indx = ['a cat sticking its head out of a cement wall' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][-2]

    indx = ['a clock is connected to the top' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][-1]


    indx = ['a girl playing tennis hits the ball back to the opponet' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][-1]

    indx = ['there is a man that is riding on a bie ' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][-2]

    indx = ['a man in suit sking down a slope with ' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][-3]

    indx = ['three small oranges being held in one hand' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][1]

    indx = ['an image of a brown cow eating grass' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][-2]

    indx = ['a small bus with the word orbit painted ' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][-2]

    indx = ['there is a bed and a windwo and two chairs on the side' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][1]

    indx = ['a couple of people are cooking in a room' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][-1]

    indx = ['a view of  city street with some traffic and buildings' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][-2]

    indx = ['a bed rooom that has a bed, small in table' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][0].replace('bed rooom', 'bedroom')

    indx = ['a woman reaching out to pet an elephants trunk that is standing on front of her' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][0].split(',')[1].replace('\'', '')[1:]

    indx = ['s bike on the grass near a fire hydrant., ' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][0].split(',')[2].replace('\'', '')[1:-1]

    indx = ['these men are having a meeting with cofee' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][4]

    indx = ['a tiny bird of a passenger side rear view ' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][0].split(',')[1][1:]

    indx = ['zebras grazing on grass in large open area near trees' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][0].split(',')[0][:-1]

    indx = ['a longboarder squats to take a tight curve' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][-2]

    indx = ['a dog happily sitting in the sand tail wagging' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][0].split(',')[0]

    indx = ['a little girl in a dress uses a hair brush on her short hair.' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][0].split(',')[1].replace('\'', '')[1:]

    indx = ['everal people riding surfboards in the ocean waves' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][-1]

    indx = ['a small bathroom features a painting of birch trees, a beveled mirror and a sink cabinet with multi-pane glass windows' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][-2]


    indx = ['a giraffe is near some green leafy branches' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][1]

    indx = ['surfer riding atop a avery large wave about to jump' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][1]


    indx = ['a clock hangs from the wall of a beat up room' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][0].split(',')[0].replace('  ',' ')

    indx = ['the scotsman train is travelling down the tracks' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][1]

    indx = ['of skiers standing at the bottom of the slope' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][-2]

    indx = ['this trees has a lot of oranges growing on it' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][-1]

    indx = ['red and blue busses are parked along the street' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][1]

    indx = ['man in skies going down a snowy slope' in x for x in caption_selected].index(True)
    caption_selected[indx] = 'man in skis going down a snowy slope'

    indx = ['bed is made with the comforter and sheets pulled back' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][-2]

    indx = ['a tad in front of the thunder of a car' in x for x in caption_selected].index(True)
    caption_selected[indx] = captions_ALL[indx][1]

    indx= ['a manchester bus with passengers is traveling down the road. ' in x for x in caption_selected].index(True)
    caption_selected[indx] = 'a manchester bus with passengers is traveling down the road.'

    indx = ['three zebras eating hay from bales inside a barn-like structure.' in x for x in caption_selected].index(True)
    caption_selected[indx] = 'three zebras eating hay from bales inside a barn-like structure.'

    indx =['individuals taking picture and posturing before a polaroid. ' in x for x in caption_selected].index(True)
    caption_selected[indx] = 'individuals taking picture and posturing before a polaroid.'

    #nsd_bench.stimulus_data.image_name[indx]

    with open(deepjuice_path+'caption_cleaned.txt', 'w') as f:
        for item in caption_selected:
            f.write("%s\n" % item)

    caption_selected = [x.lower() for x in caption_selected]
    # drop the additiona spaces in the end of each caption_selected word if it exists
    caption_selected = [x[:-1] if x[-1]==' ' else x for x in caption_selected]
    # drop the additional spaces in the beginning of each caption_selected word if it exists
    caption_selected = [x[1:] if x[0]==' ' else x for x in caption_selected]
    # drop any double spaces in the caption_selected
    caption_selected = [x.replace('  ', ' ') for x in caption_selected]
    # drop any double of more spaces in the caption_selected
    caption_selected = [re.sub(' +', ' ', x) for x in caption_selected]

    # drop period in the end of each caption_selected word if it exists
    caption_selected = [x[:-1] if x[-1]=='.' else x for x in caption_selected]
    # make sure the first character of each caption_selected is not a space
    caption_selected = [x[1:] if x[0]==' ' else x for x in caption_selected]
    # make sure the last character of each caption_selected is not a space
    caption_selected = [x[:-1] if x[-1]==' ' else x for x in caption_selected]
    # drop the period in the end of each caption_selected word if it exists
    #caption_selected = [x[:-1] if x[-1]=='.' else x for x in caption_selected]
    words_list = [x.split(' ') for x in caption_selected]
    # assert there is no empty element in words
    assert np.sum([len(x)==0 for x in words_list])==0
    # make sure no words in in words is just a space or a combination of spaces
    result = check_no_space_words(words_list)
    assert(result)  # This will print False because

    word_form = words_list
    word_id = [list(range(len(x))) for x in words_list]
    # create a counter for sentence id based on the words in each sentence
    sent_id = [list(idx * np.ones(len(x)).astype(int)) for idx, x in enumerate(words_list)]

    # flatten the list
    words = [item for sublist in words_list for item in sublist]
    # assert that no word is empty
    assert np.sum([x=='' for x in words])==0
    word_form = [item for sublist in word_form for item in sublist]
    word_id = [item for sublist in word_id for item in sublist]
    sent_id = [item for sublist in sent_id for item in sublist]

    df_extract=pd.DataFrame({'word':words,'word_id':word_id,'sent_id':sent_id,'word_form':word_form})
    p=Path(deepjuice_path,'NSD_benchmark_captions_clean_v3.pkl')
    df_extract.to_pickle(p.__str__())
