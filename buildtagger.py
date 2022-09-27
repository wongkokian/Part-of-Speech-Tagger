# python3.8 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import datetime
import numpy as np
import string

def add_tag(tag, tag_counts):
    if tag not in tag_counts:
        tag_counts[tag] = 1
    else:
        tag_counts[tag] += 1
    return tag_counts

def add_word(word, vocab_counts):
    if word not in vocab_counts:
        vocab_counts[word] = 1
    else:
        vocab_counts[word] += 1
    return vocab_counts

def add_transition(prev_tag, curr_tag, transition_counts):
    if prev_tag not in transition_counts:
        transition_counts[prev_tag] = {curr_tag: 1}
    elif curr_tag not in transition_counts[prev_tag]:
        transition_counts[prev_tag][curr_tag] = 1
    else:
        transition_counts[prev_tag][curr_tag] += 1
    return transition_counts

def add_emission(tag, word, emission_counts):
    if tag not in emission_counts:
        emission_counts[tag] = {word: 1}
    elif word not in emission_counts[tag]:
        emission_counts[tag][word] = 1
    else:
        emission_counts[tag][word] += 1
    return emission_counts

def create_table(items):
    lookup_table = {}
    curr_index = 0
    items = sorted(items.keys())
    for item in items:
        lookup_table[item] = curr_index
        curr_index += 1
    return lookup_table

def train_model(train_file, model_file):
    # write your code here. You can add functions as well.

    # Tokenize and get values for C(ti, ti-1), C(w, t), C(t)
    vocab_counts = {}
    tag_counts = {}
    transition_counts = {}
    emission_counts = {}
    file = open(train_file, 'r')
    for sentence in file.readlines():
        prev_tag = '<s>'
        tag_counts = add_tag(prev_tag, tag_counts)
        for token in sentence.split():
            word_tag = token.rsplit('/', 1)
            curr_word = word_tag[0]
            curr_tag = word_tag[1]
            vocab_counts = add_word(curr_word, vocab_counts)
            tag_counts = add_tag(curr_tag, tag_counts)
            transition_counts = add_transition(prev_tag, curr_tag, transition_counts)
            emission_counts = add_emission(curr_tag, curr_word, emission_counts)
            prev_tag = curr_tag
        tag_counts = add_tag('</s>', tag_counts)
        transition_counts = add_transition(prev_tag, '</s>', transition_counts)

    # Create lookup tables for tags and vocabulary
    tag_lookup = create_table(tag_counts)
    vocab_lookup = create_table(vocab_counts)
    disc_factor = 0.001

    # Create transition probability table P(ti|ti-1)
    transition_prob_mat = np.empty(shape=(len(tag_lookup), len(tag_lookup)))
    for prev_tag, prev_index in tag_lookup.items():
        for curr_tag, curr_index in tag_lookup.items():
            if prev_tag in transition_counts and curr_tag in transition_counts[prev_tag]:
                transition_prob_mat[prev_index][curr_index] = (transition_counts[prev_tag][curr_tag] + disc_factor) / (tag_counts[prev_tag] + disc_factor * len(tag_lookup))
            else:
                transition_prob_mat[prev_index][curr_index] = disc_factor / (tag_counts[prev_tag] + disc_factor * len(tag_lookup))

    # Create emission probability table P(wi|ti)
    emission_prob_mat = np.empty(shape=(len(tag_lookup), len(vocab_lookup)))
    for tag, tag_index in tag_lookup.items():
        for word, word_index in vocab_lookup.items():
            if tag in emission_counts and word in emission_counts[tag]:
                emission_prob_mat[tag_index][word_index] = (emission_counts[tag][word] + disc_factor) / (tag_counts[tag] + disc_factor * len(tag_lookup))
            else:
                emission_prob_mat[tag_index][word_index] = disc_factor / (tag_counts[tag] + disc_factor * len(tag_lookup))

    # Create emission probability table P(unk|ti)
    unk_emission_prob_mat = np.empty(shape=(len(tag_lookup), 1))
    for tag, tag_index in tag_lookup.items():
        unk_count = 0
        for word in vocab_lookup.keys():
            if tag in emission_counts and word in emission_counts[tag]:
                if emission_counts[tag][word] == 1:
                    unk_count += 1
        unk_emission_prob_mat[tag_index] = (unk_count + disc_factor) / (tag_counts[tag] + disc_factor * len(tag_lookup))
    
    # Create emission probability table P(caps|ti)
    caps_emission_prob_mat = np.empty(shape=(len(tag_lookup), 3))
    for tag, tag_index in tag_lookup.items():
        init_caps_count = 0
        mid_caps_count = 0
        no_caps_count = 0
        for word in vocab_lookup.keys():
            if tag in emission_counts and word in emission_counts[tag]:
                if word[0].isupper():
                    init_caps_count += emission_counts[tag][word]
                elif word[1:].isupper():
                    mid_caps_count += emission_counts[tag][word]
                else:
                    no_caps_count += emission_counts[tag][word]
        caps_emission_prob_mat[tag_index][0] = (init_caps_count + disc_factor) / (tag_counts[tag] + disc_factor * len(tag_lookup))
        caps_emission_prob_mat[tag_index][1] = (mid_caps_count + disc_factor) / (tag_counts[tag] + disc_factor * len(tag_lookup))
        caps_emission_prob_mat[tag_index][2] = (no_caps_count + disc_factor) / (tag_counts[tag] + disc_factor * len(tag_lookup))
    
    # Create emission probability table P(suffix|ti)
    suffix_emission_prob_mat = np.empty(shape=(len(tag_lookup), 6))
    for tag, tag_index in tag_lookup.items():
        s_count = 0
        es_count = 0
        ed_count = 0
        ing_count = 0
        ly_count = 0
        no_suffix_count = 0
        for word in vocab_lookup.keys():
            if tag in emission_counts and word in emission_counts[tag]:
                if word.endswith('es'):
                    es_count += emission_counts[tag][word]
                elif word.endswith('s'):
                    s_count += emission_counts[tag][word]
                elif word.endswith('ed'):
                    ed_count += emission_counts[tag][word]
                elif word.endswith('ing'):
                    ing_count += emission_counts[tag][word]
                elif word.endswith('ly'):
                    ly_count += emission_counts[tag][word]
                else:
                    no_suffix_count += emission_counts[tag][word]
        suffix_emission_prob_mat[tag_index][0] = (s_count + disc_factor) / (tag_counts[tag] + disc_factor * len(tag_lookup))
        suffix_emission_prob_mat[tag_index][1] = (es_count + disc_factor) / (tag_counts[tag] + disc_factor * len(tag_lookup))
        suffix_emission_prob_mat[tag_index][2] = (ed_count + disc_factor) / (tag_counts[tag] + disc_factor * len(tag_lookup))
        suffix_emission_prob_mat[tag_index][3] = (ing_count + disc_factor) / (tag_counts[tag] + disc_factor * len(tag_lookup))
        suffix_emission_prob_mat[tag_index][4] = (ly_count + disc_factor) / (tag_counts[tag] + disc_factor * len(tag_lookup))
        suffix_emission_prob_mat[tag_index][5] = (no_suffix_count + disc_factor) / (tag_counts[tag] + disc_factor * len(tag_lookup))

    # Create emission probability table P(punct/number|ti)
    punct_num_emission_prob_mat = np.empty(shape=(len(tag_lookup), 3))
    for tag, tag_index in tag_lookup.items():
        punct_count = 0
        num_count = 0
        none_count = 0
        for word in vocab_lookup.keys():
            if tag in emission_counts and word in emission_counts[tag]:
                if any(char.isdigit() for char in word):
                    num_count += emission_counts[tag][word]
                elif any(char in set(string.punctuation) for char in word):
                    punct_count += emission_counts[tag][word]
                else:
                    none_count += emission_counts[tag][word]
        punct_num_emission_prob_mat[tag_index][0] = (num_count + disc_factor) / (tag_counts[tag] + disc_factor * len(tag_lookup))
        punct_num_emission_prob_mat[tag_index][1] = (punct_count + disc_factor) / (tag_counts[tag] + disc_factor * len(tag_lookup))
        punct_num_emission_prob_mat[tag_index][2] = (none_count + disc_factor) / (tag_counts[tag] + disc_factor * len(tag_lookup))

    # Write lookup, transition and emission probability tables to file
    np.savez(model_file, v=vocab_lookup, t=tag_lookup, tpm=transition_prob_mat, epm=emission_prob_mat, upm=unk_emission_prob_mat, 
        cpm=caps_emission_prob_mat, spm=suffix_emission_prob_mat, ppm=punct_num_emission_prob_mat)

    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
