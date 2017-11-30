# Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from edward.models import (
    Dirichlet, Categorical, ParamMixture)


def mc_lda_model(D, S, V, K, w_obs):
    """Defines multi-channel LDA model in Edward.

    Args:
        D - Number of individuals.
        S - Number of data sources.
        V - List of vocabulary size for each data source.
        K - Number of phenotypes to model.
        w_obs - List of lists of 1D NumPy arrays containing
                tokens for each individual and data source.

    Returns:
        alpha - Dirichlet hyperparameters for prior on theta.
        beta - Dirichlet hyperparameters for prior on phi.
        theta - List of categorical parameters for individual
                phenotype distributions.
        phi - List of categorical parameters for phenotype token distributions.
        z - List of phenotype assignments for each observed token.
        w - List of observed tokens modeled as parameter mixtures.
    """

    alpha = tf.zeros(K) + 0.01

    beta, phi = [None] * S, [None] * S
    for s in range(S):
        beta[s] = tf.zeros(V[s]) + 0.01
        phi[s] = Dirichlet(concentration=beta[s],
                           sample_shape=K)

    theta = [None] * D
    w = [[None] * S for d in range(D)]
    z = [[None] * S for d in range(D)]

    for d in range(D):
        theta[d] = Dirichlet(concentration=alpha)

        for s in range(S):
            w[d][s] = ParamMixture(mixing_weights=theta[d],
                                   component_params={'probs': phi[s]},
                                   component_dist=Categorical,
                                   sample_shape=len(w_obs[d][s]),
                                   validate_args=True)

            z[d][s] = w[d][s].cat

    return alpha, beta, theta, phi, z, w


def main():
    """Empty main function."""
    return True


if __name__ == '__main__':
    main()
