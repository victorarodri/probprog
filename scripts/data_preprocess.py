# Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import pickle
import gensim
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


def __build_data_dictionary(data_type, corpus_filename, vocab_filename,
                            med_stopwords_file, docfreq_threshold):
    """Helper function for building token-id to token dictionaries.

    Args:
        data_type: The data type. Can be 'lab', 'med', or 'note'.
        corpus_filename: The filename for the input data corpus.
        vocab_filename: The filename where extracted vocabulary will be stored.
        med_stopwords_file: The filename for the medical stop words list.
        docfreq_threshold: Document-frequency threshold above which tokens are
            discarded.

    Returns:
        A gensim dictionary generated from the corpus.

    """

    # dictionary mapping tokens to token ids
    dictionary = gensim.corpora.Dictionary()

    with open(corpus_filename, 'r') as file:

        if data_type == 'note':
            # tokenizer finds and tokenizes words in text
            tokenizer = RegexpTokenizer('[a-zA-Z]+')

            # load English language stop words list
            stop_gen = set(stopwords.words('english'))

            # load medical stop words list
            with open(med_stopwords_file, 'r') as msw_file:
                stop_med = msw_file.readlines()
                stop_med = re.sub('\n', '', ';'.join(stop_med))
                stop_med = set(stop_med.split(';'))

            for line in file:
                # remove deidentification symbols in text
                text = re.sub('\[\*\*.+?\*\*\]', ' ', line).lower()

                # tokenize text and remove stop words
                tokens = tokenizer.tokenize(text)
                tokens = [t for t in tokens
                          if t not in stop_gen and t not in stop_med]

                # add tokens to dictionary
                dictionary.add_documents([tokens])

        else:  # data_type is 'lab' or 'med'
            for line in file:
                # generate tokens directly for read line
                tokens = line.strip().split('; ')

                # add tokens to dictionary
                dictionary.add_documents([tokens])

    # remove high frequency tokens from dictionary
    dictionary.filter_extremes(no_above=docfreq_threshold)
    dictionary.compactify()

    if os.path.exists(vocab_filename):
        os.remove(vocab_filename)

    # construct and save vocabulary contained in dictionary
    with open(vocab_filename, 'a') as file:
        for dit in dictionary.items():
            token = dit[1]
            file.write(token + '\n')

    return dictionary


def build_data_dictionaries(data_dir, docfreq_thresholds):
    """Helper function for building token-id to token dictionaries.

    Args:
        data_dir: Directory containing input data.
        docfreq_thresholds: Document-frequency threshold above which tokens are
            discarded.

    Returns:
        A list of gensim dictionaries generated from the corpora.

    """

    # construct necessary directory and file paths
    corpora_dir = os.path.join(data_dir, 'corpora')
    dicts_dir = os.path.join(data_dir, 'dicts')
    vocab_dir = os.path.join(data_dir, 'vocab')
    med_stopwords_file = os.path.join(data_dir, 'med_stopwords.txt')

    if not os.path.exists(dicts_dir):
        os.makedirs(dicts_dir)

    dicts = []
    data_types = ['lab', 'med', 'note']

    # build dictionaries for each data type
    for s, dt in enumerate(data_types):
        print('Building dictionary for {}.'.format(dt))

        # construct necessary file paths
        corpus_filename = os.path.join(corpora_dir, dt + '_corpus.txt')
        dict_filename = os.path.join(dicts_dir, dt + '_dict.p')
        vocab_filename = os.path.join(vocab_dir, dt + '_vocab.txt')

        if os.path.exists(dict_filename):
            os.remove(dict_filename)

        # build, save, and store dictionary
        dt_dict = __build_data_dictionary(dt,
                                          corpus_filename,
                                          vocab_filename,
                                          med_stopwords_file,
                                          docfreq_thresholds[s])

        dt_dict.save(dict_filename)

        dicts.append(dt_dict)

    return dicts


def __format_corpus(data_type, corpus_filename, dictionary,
                    med_stopwords_file):
    """Helper function for converting a raw corpus to tokenized format.

    Args:
        data_type: The data type. Can be 'lab', 'med', or 'note'.
        corpus_filename: The filename for the input data corpus.
        dictionary: Gensim dictionary for this data type.
        med_stopwords_file: The filename for the medical stop words list.

    Returns:
        A list containing the tokenized corpus.

    """

    form_corpus = []
    with open(corpus_filename, 'r') as file:

        if data_type == 'note':
            # tokenizer finds and tokenizes words in text
            tokenizer = RegexpTokenizer('[a-zA-Z]+')

            # load English language stop words list
            stop_gen = set(stopwords.words('english'))

            # load medical stop words list
            with open(med_stopwords_file, 'r') as msw_file:
                stop_med = msw_file.readlines()
                stop_med = re.sub('\n', '', ';'.join(stop_med))
                stop_med = set(stop_med.split(';'))

            for line in file:
                # remove deidentification symbols in text
                text = re.sub('\[\*\*.+?\*\*\]', ' ', line).lower()

                # tokenize text and remove stop words
                tokens = tokenizer.tokenize(text)
                tokens = [t for t in tokens
                          if t not in stop_gen and t not in stop_med]

                # apply dictionary to tokens to obtain formatted corpus
                form_corpus.append(dictionary.doc2bow(tokens))

        else:  # data_type is 'lab' or 'med'
            for line in file:
                # generate tokens directly for read line
                tokens = line.strip().split('; ')

                # apply dictionary to tokens to obtain formatted corpus
                form_corpus.append(dictionary.doc2bow(tokens))

    return form_corpus


def format_corpora(data_dir):
    """Helper function for converting a raw corpora to tokenized format.

    Args:
        data_dir: Directory containing input data.

    Returns:
        None.

    """

    corpora_dir = os.path.join(data_dir, 'corpora')
    dicts_dir = os.path.join(data_dir, 'dicts')
    form_corpora_dir = os.path.join(data_dir, 'form_corpora')
    med_stopwords_file = os.path.join(data_dir, 'med_stopwords.txt')

    if not os.path.exists(form_corpora_dir):
        os.makedirs(form_corpora_dir)

    data_types = ['lab', 'med', 'note']

    for dt in data_types:
        print('Formatting corpus for {}.'.format(dt))

        corpus_filename = os.path.join(corpora_dir, dt + '_corpus.txt')

        dict_filename = os.path.join(dicts_dir, '{}_dict.p'.format(dt))
        with open(dict_filename, 'rb') as dict_file:
            dictionary = pickle.load(dict_file)

        form_corpus = __format_corpus(dt,
                                      corpus_filename,
                                      dictionary,
                                      med_stopwords_file)

        form_corpus_filename = os.path.join(form_corpora_dir,
                                            dt + '_form_corpus.txt')

        if os.path.exists(form_corpus_filename):
            os.remove(form_corpus_filename)

        gensim.corpora.BleiCorpus.serialize(form_corpus_filename, form_corpus)


def main():
    """Main program for building dictionaries and formatting corpora"""
    data_dir = '../data'
    docfreq_thresholds = [0.999, 0.85, 0.65]
    build_data_dictionaries(data_dir, docfreq_thresholds)
    format_corpora(data_dir)


if __name__ == '__main__':
    main()
