# Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import pickle
import numpy as np


def build_toy_dataset(D, V, K):
    """Builds a toy dataset of D documents.

    Args:
        D - Number of documents.
        V - Size of vocabulary.
        K - Number of topics.

    Returns:
        w - List of 1D NumPy arrays containing word tokens for each document.
        theta - 2D NumPy array containing document topic distributions.
        phi - 2D NumPy array containing topic token distributions.
    """

    # Draw number of tokens for each document
    N = np.random.randint(low=2,
                          high=20,
                          size=D)

    # Draw topic distributions for each document
    theta = np.random.dirichlet(alpha=np.ones(K) * 0.1,
                                size=D)

    # Create topics (non-overlapping)
    phi_values = np.array([K * 1 / V] * int(V / K) + [0.0] * int(V - V / K))
    phi = np.zeros([K, V])
    for k in range(K):
        phi[k, :] = np.roll(phi_values, int(k * V / K))

    # Draw tokens for each document
    w, z = [None] * D, [None] * D
    for d in range(D):
        # Draw token topic assignments
        z[d] = np.array([np.random.choice(range(K), size=N[d],
                        p=theta[d, :])])[0]
        # Draw tokens
        w[d] = np.zeros(N[d])
        for n in range(N[d]):
            w[d][n] = np.random.choice(range(V), size=1, p=phi[z[d][n], :])

    return N, w, z, theta, phi


def load_mimic_data(data_dir):
    """Loads a dataset extracted from the MIMIC-III critical care database.

    Args:
        data_dir - String specifying directory containing input.

    Returns:
        w - List of lists of 1D NumPy arrays containing tokens for each
            individual and data source.
        dicts - List of token id to token dictionaries for all data types
    """

    data_types = ['note', 'lab', 'med']

    D = 10  # number of patients
    S = len(glob.glob(os.path.join(data_dir, 'corpora/*.txt')))

    dicts = [None] * S
    w = [[None] * S for d in range(D)]
    for s, dt in enumerate(data_types):
        dict_file = os.path.join(data_dir, 'dicts', dt + '_dict.p')

        form_corpus_file = os.path.join(data_dir, 'form_corpora',
                                        dt + '_form_corpus.txt')

        with open(dict_file, 'rb') as file:
            dicts[s] = pickle.load(file)

        with open(form_corpus_file, 'r') as file:
            for d, line in enumerate(file):
                doc_tokenids = []
                tokenid_counts = line.split(' ')[1:]

                for tic in tokenid_counts:
                    ti_c = tic.strip().split(':')
                    tokenid = float(ti_c[0])
                    count = int(ti_c[1])

                    if count == 1:
                        count += 1

                    for _ in range(count):
                        doc_tokenids.append(tokenid)

                w[d][s] = np.array(doc_tokenids)

    return w, dicts, D, S


def main():
    """Empty main function."""
    return True


if __name__ == '__main__':
    main()
