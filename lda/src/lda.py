import enum
import numpy as np
from scipy.special import gammaln, betaln

class LDA:

    def __init__(self, corpus, num_topics, num_docs, vocab_len, alpha=0.01, beta=0.01, **kwargs):
        """[summary]

        Args:
            num_topics ([type]): [description]
            num_docs ([type]): [description]
            vocab_len ([type]): [description]
            alpha (float, optional): [description]. Defaults to 0.01.
            beta (float, optional): [description]. Defaults to 0.01.
        """
        self.num_topics = num_topics
        self.num_docs = num_docs
        self.vocab_len = vocab_len
        self.alpha = alpha
        self.beta = beta
        self.theta = np.zeros([self.num_docs, self.num_topics]) + self.alpha # D x K, number of words in i'th document assigned to j'th topic, per document topic distribution
        self.phi = np.zeros([self.num_topics, self.vocab_len])  + self.beta # K x W, number of times j'th word is assigned to i'th topic, per topic word distribution
        self.nz = np.zeros(self.num_topics)  # 1 x K, total word count for each topic
        self.per_doc_word_topic_assignment = [[0] * len(doc) for doc in corpus]
    
    def initialize(self, documents):
        for d, doc in enumerate(documents):
            # topic_idxs = np.random.randint(self.n_topic, size=len(doc))  # randomly assign a topic to each word in document d
            # self.per_doc_word_topic_assignment.append(topic_idxs)  # topic assignment for each word in doc
            for w, word in enumerate(doc):
                topic_idx = np.random.randint(self.K)
                self.nz[topic_idx] += 1  # update count of total number of words assigned to j'th topic, topic count
                self.theta[d, topic_idx] += 1  # update count for current word in current document assigned to j'th topic
                self.phi[topic_idx, w] += 1  # update count for current word assigned to j'th topic
                self.per_doc_word_topic_assignment[d][w] = topic_idx
    
    def fit(self, documents, max_iter):
        for i in range(max_iter):
            for d, doc in enumerate(documents):
                for w, word in enumerate(doc):
                    topic_idx = self.per_doc_word_topic_assignment[d][w]
                    self.nz[topic_idx] -= 1  # update count of total number of words assigned to j'th topic
                    self.theta[d, topic_idx] -= 1  # update count for current word in current document assigned to j'th topic
                    self.phi[topic_idx, w] -= 1  # update count for current word assigned to j'th topic
                    self.per_doc_word_topic_assignment[d][w] = topic_idx
                    # compute posterior
                    word_topic_count = self.phi[:, w]
                    topic_count = self.nz
                    doc_topic_count = self.theta[d]

                    word_topic_ratio = (doc_topic_count + self.alpha) / (len(doc) + (self.num_topics * self.alpha))
                    topic_doc_ratio = (word_topic_count + self.beta) / (self.nz + (self.vocab_len*self.beta))
                    p_z_w = topic_doc_ratio * word_topic_ratio
                    full_cond_dist = p_z_w / np.sum(p_z_w)
                    new_topic_idx = np.random.multinomial(1, full_cond_dist).argmax()

                    self.nz[topic_idx] -= 1  # update count of total number of words assigned to j'th topic
                    self.theta[d, topic_idx] -= 1  # update count for current word in current document assigned to j'th topic
                    self.phi[topic_idx, w] -= 1  # update count for current word assigned to j'th topic
                    self.per_doc_word_topic_assignment[d][w] = new_topic_idx
            
            perplexity = np.exp(-self.log_likelihood() / self.vocab_len)  # number of token, modify?
            print(f"log likelihood: {self.log_likelihood}, perplexity: {perplexity}")

            self.phi
        
    def log_likelihood(self):
        # refer to equations 2 & 3 in Griffiths and Steyvers (2004)
        ll = 0
        # log p(w|z)
        for j in range(self.K): 
            ll += gammaln(self.vocab_len * self.beta)
            ll -= self.vocab_len * gammaln(self.beta)
            ll += np.sum(gammaln(self.phi[j] + self.beta))
            ll -= gammaln(np.sum(self.phi[j] + self.beta))
        # log p(z)
        for d, doc in enumerate(self.corpus):
            ll += gammaln(np.sum(self.alpha))
            ll -= np.sum(gammaln(self.alpha))
            ll += np.sum(gammaln(self.theta[d] + self.alpha))
            ll -= gammaln(np.sum(self.theta[d] + self.alpha))
        return ll
    



    


