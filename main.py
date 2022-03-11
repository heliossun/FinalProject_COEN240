
import numpy as np
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim_models
import collections
import seaborn as sns
import pandas as pd
from collections import OrderedDict
from sklearn.datasets import fetch_20newsgroups
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
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook, export_png
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from TransformerModel import *
import torch
import math
TopicNum = 20
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
        yield doc


def prepare_corpus(corpus):
    dictionary = corpora.Dictionary(corpus)
    #no_below (int, optional) – Keep tokens which are contained in at least no_below documents.
    #no_above (float, optional) – Keep tokens which are contained in no more than
    #no_above documents (fraction of total corpus size, not an absolute number).
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=2000)  #shape:(18846,24759)
    dictionary.save('./dictionary.dic')
    bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]
    return dictionary, bow_corpus


def visualizeLDA(model, bow_corpus,TopicNum,gnd):
    # pyLDAvis graph
    graphLDA = pyLDAvis.gensim_models.prepare(
        model, bow_corpus, dictionary=model.id2word)
    pyLDAvis.save_html(graphLDA, './graph/lda-bow.html')
    # TSNE graph
    top_dist =[]
    for d in bow_corpus:
        tmp = {i:0 for i in range(TopicNum)}
        tmp.update(dict(model[d]))
        vals = list(OrderedDict(tmp).values())
        top_dist += [np.array(vals)]
    top_dist = np.array(top_dist)

    # tSNE Dimension Reduction
    tsne_model = TSNE(n_components=2)
    tsne_lda = tsne_model.fit_transform(top_dist)
    print(tsne_lda)
    sns.set_theme()
    topic_num=np.argmax(top_dist, axis=1)
    output_notebook()
    mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
    plot = figure(title="t-SNE Clustering of {} LDA Topics".format(TopicNum),
                plot_width=900, plot_height=700)
    plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
    output_file("./graph/LDA_TSNE.html")
    show(plot)
    export_png(plot, filename="./graph/LDA_TSNE.png")

def visualizeWrd2V(model):
    num_dimensions = 3
    labels = []
    tokens = []
    colors = cm.rainbow(np.linspace(0, 1, 1))
    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, init='pca', random_state=0)
    vectors = tsne.fit_transform(tokens)
    fig=plt.figure(figsize=(12,12))
    ax=Axes3D(fig)
    plt.scatter(vectors[:,0],vectors[:,1],vectors[:,2],c=colors,alpha=1,label=labels)

    plt.savefig(f"./workdir/Word2Vec_TSNE.png")

def KMeansCluster(X, gnd, true_k,transformer_type):

    print('>>>>>>>>>>>>>> k-means clustering begin <<<<<<<<<<<<')
    km = KMeans(n_clusters=true_k,
                    init='k-means++',
                    max_iter=30, n_init=1)
    km.fit(X)
    print('NMI score: ', nmi_score(gnd,km.labels_))
    clusters = km.labels_.tolist()
    y_km = km.fit_predict(X)
    plt.figure(figsize=(12,12))
    for i in range (0,true_k):
        plt.scatter(X[y_km==i, 0], X[y_km==i, 1],
            s=20,
            marker='o',
            alpha=.8,
            label='cluster ' + str(i)
                    )
    plt.legend(fancybox=True, framealpha=0.5,scatterpoints=1)
    plt.grid()
    print('>>>>>>>>>>>>>> k-means clustering end <<<<<<<<<<<<')
    plt.savefig(f"./workdir/{transformer_type}_KMeansCluster.png")

def TSNEVisualize(new_corpus,transformer_type, gnd,semantic_labels):
    print('>>>>>>>>>>>>>> TSNE visualization generating <<<<<<<<<<<<')
    tsne_model = TSNE(n_components=2)
    tsne_X = tsne_model.fit_transform(new_corpus)
    #print(len(tsne_X))
    k=len(set(gnd))
    sns.set_theme()
    fig = plt.figure()
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    for i in range(k):
        ax.scatter(tsne_X[gnd==i,0], tsne_X[gnd==i,1], alpha=.8, label=semantic_labels[i])
    ax.legend(fancybox=True, framealpha=0.5)
    print('>>>>>>>>>>>>>> TSNE Visualization saved <<<<<<<<<<<<')
    fig.savefig(f"./workdir/{transformer_type}_TSNE.png", dpi=fig.dpi)

def backbone(transformer_type, corpus, dictionary, bow_corpus, model_path=''):
    # initialize feature extraction model, use model to transform BOW to other representation
    #finally return the new_corpus value [[],[],[]] shape : docNum * feature_dim
    vector_size = 100
    min_count = 3
    epochs = 10
    new_corpus = []
    if transformer_type == 'TF_IDF':
        if model_path != '':
            model = models.TfidfModel.load(model_path)
        else:
            model = models.TfidfModel(bow_corpus)
            model.save('./workdir/tf_idf.py')
        new_corpus = model[bow_corpus]  # corpus tfidf
        num_docs = len(new_corpus)
        num_terms = len(dictionary.keys())
        #transoform TD-IDF-BOW to vector representation
        print('new corpus shape is : ',len(new_corpus),', ',num_terms)
        new_corpus = np.transpose(corpus2dense(new_corpus, num_terms, num_docs))
        return new_corpus
    elif transformer_type == 'BOW':
        num_docs = len(bow_corpus)
        num_terms = len(dictionary.keys())
        #transoform TD-IDF-BOW to vector representation
        print('new corpus shape is : ',len(bow_corpus),', ',num_terms)
        new_corpus = np.transpose(corpus2dense(bow_corpus, num_terms, num_docs))
        return new_corpus
    elif transformer_type == 'LDA':
        # Set training parameters.
        TopicNum = 10
        eval_every = 5

        model = LdaModel(
            corpus=bow_corpus,
            id2word=dictionary,
            alpha='auto',
            num_topics=TopicNum,
            eval_every=eval_every
        )

        # visualizeLDA(model, bow_corpus, TopicNum)
        return model

    elif transformer_type == 'D2V':
        train_corpus = [models.doc2vec.TaggedDocument(
            token, [i])for i, token in enumerate(corpus)]
        # print(train_corpus[:2])
        if model_path != '':
            model = models.doc2vec.Doc2Vec.load(model_path)
        else:

            # vector size can be set from 100-300
            model = models.doc2vec.Doc2Vec(
                vector_size=vector_size, min_count=min_count, epochs=epochs)
            model.build_vocab(train_corpus)
            print('>>>>>>>>>>>>>> start training doc2vec <<<<<<<<<<<<')
            model.train(
                train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
            print('>>>>>>>>>>>>>> model saved <<<<<<<<<<<<')
            model.save(f'./workdir/d2v_{vector_size}_{min_count}_{epochs}.py')
            #print('>>>>>>>>>>>>>> assessing the model <<<<<<<<<<<<')
            #assesModel(model, train_corpus)
        new_corpus = model.docvecs.doctag_syn0
        return new_corpus
    else: #Word2Vec
        if model_path != '':
            model = models.Word2Vec.load(model_path)
        else:
            print('>>>>>>>>>>>>>> start training doc2vec <<<<<<<<<<<<')
            model = models.Word2Vec(corpus,size=vector_size,min_count=min_count, iter=epochs)
            model.save(f'./workdir/w2v_{vector_size}_{min_count}_{epochs}.py')
            print('>>>>>>>>>>>>>> model saved <<<<<<<<<<<<')
        return model

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
def term_f_dist(rawdata,newdata):
    dist=[]
    for i in range(len(rawdata)):
        if len(rawdata[i]) ==0 or len(newdata[i]) ==0:
            #print('intdex:', i, ' has zeros: raw val ',len(rawdata[i]),'-- new val ',len(newdata[i]))
            dist.append(1)
        else:
            dist.append(len(newdata[i])/len(rawdata[i]))
    fig = plt.figure()
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    ax.scatter([i for i in range(len(rawdata))],dist)
    ax.legend(fancybox=True, framealpha=0.5)
    ax.set_ylabel('term frequency(len(newtoken)/len(orgtoken))')
    ax.set_xlabel('document(id)')
    ax.set_title('Term Frequency Distribution')
    fig.savefig(f"./workdir/term_f_dist.png", dpi=fig.dpi)

def featureEmbedding(transformer_type, model_path):
    # load dataset
    dataset = fetch_20newsgroups(
        subset='all', shuffle=False, remove=('headers', 'footers', 'quotes'))
    corpus = dataset.data  # save as the raw docs
    corpus1 = list(pre_processing(corpus))
    term_f_dist(corpus, corpus1)
    print(f'>>>>>>>>>dist plot saved<<<<<<<<<')
    # labels for clustering evaluation or supervised tasks length is 18846
    gnd = dataset.target

    true_k = np.unique(gnd).shape[0]
    semantic_labels = list(range(true_k))
    dictionary, bow_corpus = prepare_corpus(corpus1)
    new_corpus = bow_corpus
    if(transformer_type == "TF_IDF" or transformer_type == "D2V"):
        #generate corpus of 4 different methods(TF-IDF, LDA, Doc2Vec, Word2Vec)
        new_corpus = backbone(transformer_type, corpus1,
                            dictionary, bow_corpus, model_path)
        print(len(new_corpus),', ',len(new_corpus[0]))
        #TSNEVisualize(new_corpus,transformer_type, gnd,semantic_labels)
        #KMeansCluster(new_corpus, gnd, true_k,transformer_type)
        return new_corpus, gnd
    #TSNEvisualize (TF-IDF, Doc2Vec, Word2Vec)
    elif transformer_type == 'LDA':
        model = backbone(transformer_type, corpus1,
                            dictionary, bow_corpus, model_path)
        visualizeLDA(model, bow_corpus,TopicNum,gnd)
        #this new_corpus is LDA model actually
        #  and get the topic distribution (as features) for each document
        for bow in bow_corpus:
            t = new_corpus.get_document_topics(bow)
            print(t)
    elif transformer_type == "BOW":
        new_corpus = backbone(transformer_type, corpus1,
                            dictionary, bow_corpus, model_path)
        #TSNEVisualize(new_corpus,transformer_type, gnd,semantic_labels
        KMeansCluster(new_corpus, gnd, true_k,transformer_type)


    else:   #transformer_type == 'W2V'
        model = backbone(transformer_type, corpus1,
                            dictionary, bow_corpus, model_path)
        visualizeWrd2V(model)
        #this new_corpus is W2V model actually
def data_loader(transformer_type, model_path):
    docRepresent, label = featureEmbedding(transformer_type, model_path)
    dataset = [torch.tensor(tuple(doc,gnd))for doc in docRepresent for gnd in label]
    train = math.ceil(len(dataset*0.7))
    validate=math.ceil(len(dataset*0.2))
    print(dataset[0])
def batchify(data, batch_size):

def get_batch(source,i):

def train(transformer_type, model_path):
    data_loader(transformer_type, model_path)






if __name__ == '__main__':
    transformer_type = 'D2V'    #LDA/ D2V/ TF_IDF/BOW/W2V/BOW
    model_path = './workdir/d2v_100_3_40.py'
    train(transformer_type, model_path)
