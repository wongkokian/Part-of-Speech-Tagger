# python3.8 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import datetime
import numpy as np
import string

def get_caps_prob(word, mat):
    if word[0].isupper():
        return mat[0]
    elif word[1:].isupper():
        return mat[1]
    else:
        return mat[2]

def get_suffix_prob(word, mat):
    if word.endswith('es'):
        return mat[1]
    elif word.endswith('s'):
        return mat[0]
    elif word.endswith('ed'):
        return mat[2]
    elif word.endswith('ing'):
        return mat[3]
    elif word.endswith('ly'):
        return mat[4]
    else:
        return mat[5]

def get_punct_num_prob(word, mat):
    if any(char.isdigit() for char in word):
        return mat[0]
    elif any(char in set(string.punctuation) for char in word):
        return mat[1]
    else:
        return mat[2]

def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.
    test = open(test_file, 'r')
    out = open(out_file, 'w')
    archive = np.load(model_file + '.npz', allow_pickle=True)
    vocab_lookup = archive['v'][()]
    tag_lookup = archive['t'][()]
    transition_prob_mat = archive['tpm']
    emission_prob_mat = archive['epm']
    unk_emission_prob_mat =archive['upm']
    caps_emission_prob_mat =archive['cpm']
    suffix_emission_prob_mat =archive['spm']
    punct_num_emission_prob_mat =archive['ppm']

    for sentence in test.readlines():
        result = ''
        tokenized_sentence = sentence.split()
        prev_tag_index = tag_lookup['<s>']
        prev_prob = 1
        for word in tokenized_sentence:
            max_prob_tag = ''
            max_prob = 0
            for curr_tag, curr_tag_index in tag_lookup.items():
                transition_prob = transition_prob_mat[prev_tag_index][curr_tag_index]
                if word in vocab_lookup:
                    emission_prob = emission_prob_mat[curr_tag_index][vocab_lookup[word]]
                    curr_prob = prev_prob * transition_prob * emission_prob
                else:
                    caps_prob = get_caps_prob(word, caps_emission_prob_mat[curr_tag_index])
                    suffix_prob = get_suffix_prob(word, suffix_emission_prob_mat[curr_tag_index])
                    punct_num_prob = get_punct_num_prob(word, punct_num_emission_prob_mat[curr_tag_index])
                    emission_prob = unk_emission_prob_mat[curr_tag_index][0] * caps_prob * suffix_prob * punct_num_prob
                    curr_prob = prev_prob * transition_prob * emission_prob
                if curr_prob >= max_prob:
                    max_prob = curr_prob
                    max_prob_tag = curr_tag
            result += (word + '/' + max_prob_tag + ' ')
            prev_tag_index = tag_lookup[max_prob_tag]
            prev_prob = max_prob
        out.write(result.strip() + '\n')
    out.close()

    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
