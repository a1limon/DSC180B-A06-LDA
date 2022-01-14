import numpy as np

try:
    from mcmc_logreg import mcmc_log_reg
except ImportError:
    from mcmc.src.mcmc_logreg import mcmc_log_reg

import warnings
warnings.filterwarnings('ignore')

def run_mh_mcmc(X, y, **kwargs):
    beta_priors = np.repeat(kwargs['beta_prior_init'], X.shape[1]) 
    stddevs_priors = np.repeat(kwargs['stddev_prior_init'], X.shape[1])
    stddevs_proposal_dist = np.repeat(kwargs['stddev_prop_init'], X.shape[1])
    solver = mcmc_log_reg()
    solver.mh_mcmc(
        y,
        X,
        beta_priors, 
        stddevs_priors,
        stddevs_proposal_dist, 
        kwargs['num_steps'],
        random_seed=6
    )
    solver.beta_distr = solver.raw_beta_distr
    solver.fit('mean')
    pred = solver.predict(X)
    pred = [1 if i > .5 else 0 for i in pred.flatten()]
    y = y.flatten()
    correct = pred == y
    print("accuracy: ", sum(correct) / len(correct))
    