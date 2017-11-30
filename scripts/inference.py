# Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import edward as ed
import tensorflow as tf
from edward.models import Empirical


def mc_lda_inference_setup(T, D, S, V, K, w_obs, model_vars):
    """Defines data, latent, and proposal variables for multi-channel
    LDA model.

    Args:
        T - Number of samples to draw during inference.
        D - Number of individuals.
        S - Number of data sources.
        V - List of vocabulary size for each data source.
        K - Number of phenotypes to model.
        w_obs - List of lists of 1D NumPy arrays containing
                tokens for each individual and data source.
        model_vars - List containing random variables in mulit-channel
                     LDA: [theta, phi, z, w].

    Returns:
        latent_vars_dict - Dictionary mapping model latent variables to
                           empirical random variables (e.g. phi to qphi).
        proposal_vars_dict - Dictionary mapping model late variables to
                             complete conditionals.
        data_dict - Dictionary mapping model observed random variables to
                    observed data.
    """

    total_time = time.time()  # track total time cost

    theta = model_vars[0]
    phi = model_vars[1]
    z = model_vars[2]
    w = model_vars[3]

    # Data variables
    data_dict = {}
    for d in range(D):
        for s in range(S):
            data_dict[w[d][s]] = w_obs[d][s]

    # Latent variabless
    latent_vars_dict = {}

    qphi = [None] * S
    for s in range(S):
        qphi[s] = Empirical(tf.Variable(tf.zeros([T, K, V[s]])))
        latent_vars_dict[phi[s]] = qphi[s]

    qtheta = [None] * D
    qz = [[None] * S for d in range(D)]
    for d in range(D):
        qtheta[d] = Empirical(tf.Variable(tf.ones([T, K]) / K))
        latent_vars_dict[theta[d]] = qtheta[d]

        for s in range(S):
            N = len(w_obs[d][s])

            qz[d][s] = Empirical(tf.Variable(tf.zeros([T, N], dtype=tf.int32)))
            latent_vars_dict[z[d][s]] = qz[d][s]
    print()

    # Proposal variables
    proposal_vars_dict = {}

    phi_cond = [None] * S
    for s in range(S):
        it_time = time.time()

        print('Building proposals for phi: {} of {}'.format(s + 1, S))
        phi_cond[s] = ed.complete_conditional(phi[s])
        proposal_vars_dict[phi[s]] = phi_cond[s]

        end = time.time()
        print('Total time: {:.2f}s, '.format(end - total_time),
              'Iteration time: {:.2f}s\n'.format(end - it_time))
    print()

    theta_cond = [None] * D
    z_cond = [[None] * S for d in range(D)]
    for d in range(D):
        it_time = time.time()
        print('Building proposals for theta, z: {} of {}'.format(d + 1, D))

        theta_cond[d] = ed.complete_conditional(theta[d])
        proposal_vars_dict[theta[d]] = theta_cond[d]

        for s in range(S):
            z_cond[d][s] = ed.complete_conditional(z[d][s])
            proposal_vars_dict[z[d][s]] = z_cond[d][s]

        end = time.time()
        print('Total time: {:.2f}s, '.format(end - total_time),
              'Iteration time: {:.2f}s\n'.format(end - it_time))

    return latent_vars_dict, proposal_vars_dict, data_dict


def mc_lda_inference(T, latent_vars_dict, proposal_vars_dict, data_dict):
    """Runs Gibbs sampler for multi-channel LDA model.

    Args:
        T - Number of samples to draw during inference.
        latent_vars_dict - Dictionary mapping model latent variables to
                           empirical random variables (e.g. phi to qphi).
        proposal_vars_dict - Dictionary mapping model late variables to
                             complete conditionals.
        data_dict - Dictionary mapping model observed random variables to
                    observed data.

    Returns:
        latent_vars_dict - Dictionary mapping model latent variables to
                           empirical random variables (e.g. phi to qphi).

    """
    # Inference procedure w/Gibbs sampling
    inference = ed.Gibbs(latent_vars=latent_vars_dict,
                         proposal_vars=proposal_vars_dict,
                         data=data_dict)

    inference.initialize(n_iter=T, n_print=10, logdir='log')

    tf.global_variables_initializer().run()

    for n in range(inference.n_iter):
        info_dict = inference.update()
        inference.print_progress(info_dict)

    inference.finalize()

    return latent_vars_dict


def main():
    """Empty main function."""
    return True


if __name__ == '__main__':
    main()
