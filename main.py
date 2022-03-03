
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
import collections

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

def backbone(transformer_type, corpus, bow_corpus, model_path=''):
    #initialize feature extraction model
    model = None
    new_corpus = None
    if transformer_type == "CVS":
        pass
    elif transformer_type == "TF_IDF":
        model = models.TfidfModel(bow_corpus)
        new_corpus = [model[dictionary.doc2bow(doc)]for doc in corpus]
    elif transformer_type == "LDA":
        pass
    else:
        train_corpus = [models.doc2vec.TaggedDocument(token, [i])for i, token in enumerate(corpus)]
        if model_path != '':
            model = models.doc2vec.Doc2Vec.load(model_path)
        else:
            vector_size = 100
            min_count = 3
            epochs = 10
            model = models.doc2vec.Doc2Vec(vector_size = vector_size,min_count=min_count,epochs=epochs)#vector size can be set from 100-300
            model.build_vocab(train_corpus)
            print('>>>>>>>>>>>>>> start training doc2vec <<<<<<<<<<<<')
            model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
            print('>>>>>>>>>>>>>> model saved <<<<<<<<<<<<')
            model.save(f'./workdir/w2v_{vector_size}_{min_count}_{epochs}.py')
            #print('>>>>>>>>>>>>>> assessing the model <<<<<<<<<<<<')
            #assesModel(model, train_corpus)
        new_corpus=[model[train_corpus[doc_id].words]for doc_id in range(len(train_corpus))]

def assesModel(model, train_corpus):
    ranks = []
    second_ranks = []
    for doc_id in range(len(train_corpus)):
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)
        second_ranks.append(sims[1])
    counter = collections.Counter(ranks)
    print(counter)

def train(transformer_type, model_path):
    # load dataset
    dataset = fetch_20newsgroups(subset='all', shuffle=False, remove=('headers', 'footers', 'quotes'))
    corpus = dataset.data  # save as the raw docs
    corpus1 = list(pre_processing(corpus))
    labels = dataset.target  # labels for clustering evaluation or supervised tasks
    dictionary, bow_corpus = prepare_corpus(corpus1)
    print(bow_corpus[0])
    new_corpus = backbone(transformer_type, corpus1, bow_corpus,model_path)
    print(new_corpus[0])


if __name__ == '__main__':
    train('d2v','./workdir/w2v_100_3_40.py')
