import numpy as np
from numpy import median, mean
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.stats import norm
from tqdm import tqdm


class mcmc_log_reg:
    
    def __init__(self):
        # initialize values
        self.raw_beta_distr = np.empty(1)  # 'true' distribution of beta coefficients
        self.beta_distr = np.empty(1)
        self.beta_hat = np.empty(1)  # estimates of beta
    
    def inv_logit(self, beta, x):
        """[summary]

        Args:
            beta ([type]): [description]
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        num = np.exp(np.matmul(x,beta.reshape((-1,1)),dtype=np.float128))
        denom = 1 + (num)
        p_hat = num/denom
        return p_hat
    
    def log_prior(self, beta, prior_means, prior_stds):
        """[summary]

        Args:
            beta ([type]): [description]
            prior_means ([type]): [description]
            prior_stds ([type]): [description]

        Returns:
            [type]: [description]
        """
        prior_mu = prior_means.reshape((-1, 1))
        prior_sigma = prior_stds.reshape((-1, 1))
        return np.sum(norm.logpdf(beta, loc=prior_mu, scale=prior_sigma))
    
    def log_likelihood(self, y, x, beta):
        """[summary]

        Args:
            y ([type]): [description]
            x ([type]): [description]
            beta ([type]): [description]

        Returns:
            [type]: [description]
        """
        p = self.inv_logit(beta.reshape((-1, 1)), x)
        p_ = 1 - p
        p = np.where(p <=0, 0, p)
        p_ = np.where(p_ <= 0, 0, p_)
        epsilon = 1e-5 
        return np.sum(y * np.log(p+epsilon) + (1-y)*np.log(p_+epsilon))
    
    def log_posterior(self, y, X, beta, prior_means, prior_stds):
        """[summary]

        Args:
            y ([type]): [description]
            X ([type]): [description]
            beta ([type]): [description]
            prior_means ([type]): [description]
            prior_stds ([type]): [description]

        Returns:
            [type]: [description]
        """
        prior = self.log_prior(beta, prior_means, prior_stds)
        likelihood = self.log_likelihood(y, X, beta)
        return prior + likelihood
    
    def mh_mcmc(self, y, X, beta_priors, prior_stds, proposal_stds, num_steps,random_seed):
        """[summary]

        Args:
            y ([type]): [description]
            X ([type]): [description]
            beta_priors ([type]): [description]
            prior_stds ([type]): [description]
            proposal_stds ([type]): [description]
            num_steps ([type]): [description]
            random_seed ([type]): [description]
        """
        np.random.seed(random_seed)
        beta_idx = [k for k in range(len(beta_priors))]
        # estimates, initialized with beta_priors, has shape: [# of beta coefficients, # of steps]
        beta_hat = np.repeat(beta_priors, num_steps+1)  # repeats beta_priors num_steps+1 times
        beta_hat = beta_hat.reshape((beta_priors.shape[0], num_steps+1))
        
        for i in tqdm(range(1, num_steps + 1)):
            for j in beta_idx:
                # draw proposal from the proposal pdf, scalar
                proposal_beta_j = beta_hat[j, i-1] + norm.rvs(loc=0, \
                                                              scale=proposal_stds[j], \
                                                              size=1)
                proposal_beta_j = norm.rvs(loc= beta_hat[j, i-1], \
                                                              scale=proposal_stds[j], \
                                                              size=1)
                # vector of all beta estimates from most recent sample, shape [# beta coeff, 1]
                beta_curr = beta_hat[:, i-1].reshape((-1, 1))
                # at j'th index of most recent sample, update old proposal with a new proposal 
                beta_prop = np.copy(beta_curr)
                beta_prop[j, 0] = proposal_beta_j
                # calculate the posterior probability of the proposed beta, numerator
                log_p_proposal = self.log_posterior(y, X, beta_prop, 
                                                    beta_curr, prior_stds)
                # calculate the posterior probability of the current beta, denominator
                log_p_previous = self.log_posterior(y, X, beta_curr, 
                                                    beta_curr, prior_stds)
                
                # difference of log probabiltites
                log_diff = log_p_proposal - log_p_previous
                
                # accept-reject step
                log_r = np.log(np.random.random())
                if log_diff > log_r:
                    beta_hat[j, i] = proposal_beta_j
                else:
                    beta_hat[j, i] = beta_hat[j, i-1]
        
        self.raw_beta_distr = beta_hat
    

    def fit(self, method):
        """[summary]

        Args:
            method (str, optional): [description].
        """
        if method == 'median':
            beta_hat = median(self.beta_distr, axis=1).reshape((-1,1))
        elif method == 'mean':
            beta_hat = mean(self.beta_distr, axis=1).reshape((-1,1))

        self.beta_hat = beta_hat
        
    def predict(self, X, prob=True):
        """[summary]

        Args:
            X ([type]): [description]
            prob (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        # outputs predicted probabilities
        if prob:
            predictions = self.inv_logit(self.beta_hat, X)
        # outputs predicted log-odds
        else:
            predictions = np.matmul(X, self.beta_hat)
            
        return predictions
    
    def plot_param_hist(self, beta_distr):
        for i in range(beta_distr.shape[0]):
            plt.hist(beta_distr[i,:])
            mean = beta_distr[i,:].mean()
            plt.axvline(mean, color="red")
            min_ylim, max_ylim = plt.ylim()
            plt.title(f"Distribution of beta_{i} parameter")
            plt.xlabel('beta')
            plt.ylabel(f'Probability density beta_{i}')
            plt.text(mean*1.3, max_ylim*0.7, 'Mean: {:.2f}'.format(mean))
            # plt.savefig(f"data/histogram_beta_{i}.png")
            plt.show()

    def plot_param_trace(self, beta_distr):
        for i in range(beta_distr.shape[0]): 
            plt.plot(beta_distr[i,:])
            plt.title(f"Trace plot of beta_{i} parameter")
            plt.xlabel('iterations')
            plt.ylabel(f'beta_{i}')
            # plt.savefig(f"data/trace_plot_beta_{i}.png")
            plt.show()
