
import numpy as np
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim_models
import collections
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from stop_words import get_stop_words
from gensim import corpora
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string
from gensim import models
from gensim.matutils import corpus2dense, corpus2csc
from gensim.models import LdaModel
from pprint import pprint
from copy import deepcopy


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
    # freqs = defaultdict(int)
    # for doc in corpus1:
    #     for w in doc:
    #         freqs[w] += 1

    # preprocessing dictionary
    dictionary = corpora.Dictionary(corpus1)
    # low_tf_tokens = [w for w in freqs if freqs[w]<=3]
    # remove_ids = [dictionary.token2id[w] for w in low_tf_tokens]
    # dictionary.filter_tokens(remove_ids)
    # dictionary.compactify()  # remove gaps in id sequence after words that were removed
    dictionary.filter_extremes(no_below=5, no_above=0.1)
    dictionary.save('./vocabs/vocab')
    bow_corpus = [dictionary.doc2bow(doc) for doc in corpus1]
    return dictionary, bow_corpus


def visualizeLDA(model, bow_corpus):
    graphLDA = pyLDAvis.gensim_models.prepare(
        model, bow_corpus, dictionary=model.id2word)
    pyLDAvis.save_html(graphLDA, './graph/lda-bow.html')
    return
    return


def backbone(transformer_type, corpus, dictionary, bow_corpus, model_path=''):
    # initialize feature extraction model
    new_corpus = []
    if transformer_type == 'CVS':
        pass
    elif transformer_type == 'TF_IDF':
        if model_path != '':
            model = models.TfidfModel.load(model_path)
        else:
            model = models.TfidfModel(bow_corpus)
            model.save('./workdir/tf_idf.py')
        new_corpus = model[bow_corpus]  # corpus tfidf
        num_docs = len(new_corpus)
        num_terms = len(dictionary.keys())
        print('length of fict keys', num_terms)
        new_corpus = corpus2dense(new_corpus, num_terms, num_docs)

    elif transformer_type == 'LDA':
        # Set training parameters.
        num_topics = 50
        eval_every = 5

        model = LdaModel(
            corpus=bow_corpus,
            id2word=dictionary,
            alpha='auto',
            num_topics=num_topics,
            eval_every=eval_every
        )

        visualizeLDA(model, bow_corpus)

    else:
        train_corpus = [models.doc2vec.TaggedDocument(
            token, [i])for i, token in enumerate(corpus)]
        # print(train_corpus[:2])
        if model_path != '':
            model = models.doc2vec.Doc2Vec.load(model_path)
        else:
            vector_size = 100
            min_count = 3
            epochs = 10
            # vector size can be set from 100-300
            model = models.doc2vec.Doc2Vec(
                vector_size=vector_size, min_count=min_count, epochs=epochs)
            model.build_vocab(train_corpus)
            print('>>>>>>>>>>>>>> start training doc2vec <<<<<<<<<<<<')
            model.train(
                train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
            print('>>>>>>>>>>>>>> model saved <<<<<<<<<<<<')
            model.save(f'./workdir/w2v_{vector_size}_{min_count}_{epochs}.py')
            #print('>>>>>>>>>>>>>> assessing the model <<<<<<<<<<<<')
            #assesModel(model, train_corpus)
        new_corpus = [model.infer_vector(doc)for doc in corpus]
    return new_corpus


def assesModel(model, train_corpus):
    ranks = []
    second_ranks = []
    for doc_id in range(len(train_corpus)):
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        sims = model.docvecs.most_similar(
            [inferred_vector], topn=len(model.docvecs))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)
        second_ranks.append(sims[1])
    counter = collections.Counter(ranks)
    # print(counter)


def docClustering(transformer_type, model_path):
    # load dataset
    dataset = fetch_20newsgroups(
        subset='all', shuffle=False, remove=('headers', 'footers', 'quotes'))
    corpus = dataset.data  # save as the raw docs
    corpus1 = list(pre_processing(corpus[:1000]))
    # labels for clustering evaluation or supervised tasks length is 18846
    gnd = dataset.target
    #semantic_labels = dataset.target_names
    dictionary, bow_corpus = prepare_corpus(corpus1)
    new_corpus = backbone(transformer_type, corpus1,
                          dictionary, bow_corpus, model_path)

    if(transformer_type != 'LDA'):
        true_k = np.unique(gnd).shape[0]
        # print(true_k)
        km = KMeans(n_clusters=true_k,
                    init='k-means++',
                    max_iter=5, n_init=10)
        km.fit(new_corpus)
        clusters = km.labels_.tolist()
        print(len(clusters))
        sns.set_theme()
        fig = plt.figure()
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        for i in range(len(new_corpus)):
            ax.scatter(new_corpus[i, 0], new_corpus[i, 1],
                       alpha=.8, label=clusters[i])
        ax.legend(fancybox=True, framealpha=0.5)
        fig.savefig('./temp.png', dpi=fig.dpi)


if __name__ == '__main__':
    docClustering('TF_IDF', '')
    # docClustering('LDA','')
