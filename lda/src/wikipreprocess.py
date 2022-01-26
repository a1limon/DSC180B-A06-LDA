import os
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

class WikiPreprocess:
    def __init__(self, pos, min_token_len, no_below, no_above, keep_n):
        self.pos = pos  # part of speech tag
        self.min_token_len = min_token_len  # min length token to filter out when preprocessing
        self.no_below = no_below  # filter out tokens that appear in less than `no_below` documents
        self.no_above = no_above  # filter out tokens that appear in more than `no_above` of corpus, a fraction
        self.keep_n = keep_n  # keep only `keep_n` most frequent tokens

    def lemmatize(self, token: str) -> str:
        """Lemmatizes token using WordNet Lemmatizer

        Args:
            token (str): Article token

        Returns:
            str: Lemmatized token
        """
        return WordNetLemmatizer().lemmatize(token, pos=self.pos)

    def preprocess_article(self, text: str) -> list:
        """Preprocesses an article.
        
        Args:
            text (str): Wikipedia article

        Returns:
            list: List of preprocessed article tokens
        """
        tokens = gensim.utils.simple_preprocess(text, min_len=self.min_token_len)
        preprocessed_article = [self.lemmatize(token) for token in tokens if token not in STOPWORDS]
        return preprocessed_article

    def create_dictionary(self, preprocessed_articles: list) -> gensim.corpora.Dictionary:
        return gensim.corpora.Dictionary(preprocessed_articles)

    def filter_out(self, preprocessed_articles: list) -> gensim.corpora.Dictionary:
        """Initializes a Dictionary and filters out tokens by their frequency.

        Args:
            preprocessed_articles (list): Preprocessed articles

        Returns:
            gensim.corpora.Dictionary: Filtered dictionary
        """
        dictionary = self.create_dictionary(preprocessed_articles)
        dictionary.filter_extremes(self.no_below, self.no_above, self.keep_n)
        return dictionary

    def articles_to_bow(self, filtered_dict: gensim.corpora.Dictionary, preprocessed_articles:list) -> list:
        """Converts articles into the bag-of-words (BoW) format = list of (token_id, token_count) tuples.

        Args:
            filtered_dict (gensim.corpora.Dictionary): Filtered dictionary
            preprocessed_articles (list): Preprocessed articles, list of (token_id, token_count) tuples for each article

        Returns:
            list: preprocessed articles in BoW format
        """
        preprocessed_bow_articles = [filtered_dict.doc2bow(article) for article in preprocessed_articles]
        return preprocessed_bow_articles
    