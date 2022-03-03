
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string
from pprint import pprint
from copy import deepcopy
from gensim import models


def pre_processing(docs):
    tokenizer = RegexpTokenizer(r"\w+(?:[-'+]\w+)*|\w+")
    en_stop = get_stop_words('en')
    for doc in docs:
        raw_text = doc.lower()
        # tokenization
        tokens_text = tokenizer.tokenize(raw_text)
        # remove stopwords
        stopped_tokens_text = [i for i in tokens_text if not i in en_stop]
        # remoce digis and one-charcter word
        doc = [token for token in stopped_tokens_text if not token.isnumeric()]
        doc = [token for token in stopped_tokens_text if len(token) > 1]
        # you could always add some new preprocessing here
        yield doc

def prepare_corpus(corpus1):

    # remove words that appear only
    freqs = defaultdict(int)
    for doc in corpus1:
        for w in doc:
            freqs[w] += 1

    #preprocessing dictionary
    dictionary = corpora.Dictionary(corpus1)
    low_tf_tokens = [w for w in freqs if freqs[w]<=3]
    remove_ids = [dictionary.token2id[w] for w in low_tf_tokens]
    dictionary.filter_tokens(remove_ids)
    dictionary.compactify()  # remove gaps in id sequence after words that were removed
    dictionary.save('your_savepath')

    bow_corpus = [dictionary.doc2bow(doc) for doc in corpus1]
    return dictionary, bow_corpus

def backbone(transformer_type, bow_corpus):
    #initialize feature extraction model
    model = None
    if transformer_type == "CVS":
        pass
    elif transformer_type == "TF_IDF":
        model = models.TfidfModel(bow_corpus)
    elif transformer_type == "LDA":
        pass
    else:
        pass
    return model
def train():
    # load dataset
    dataset = fetch_20newsgroups(subset='all', shuffle=False, remove=('headers', 'footers', 'quotes'))
    corpus = dataset.data  # save as the raw docs
    corpus1 = list(pre_processing(corpus))
    labels = dataset.target  # labels for clustering evaluation or supervised tasks
    dictionary, bow_corpus = prepare_corpus(corpus1)
    print(bow_corpus[0])
    transform_model = backbone("TF_IDF", bow_corpus)
    bow_corpus = [dictionary.doc2bow(doc) for doc in corpus1]
    new_corpus = [transform_model[dictionary.doc2bow(doc)]for doc in corpus1]
    print(new_corpus[0])


if __name__ == '__main__':
    train()
