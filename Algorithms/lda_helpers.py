import pandas as pd
import gensim
import pyLDAvis.gensim_models
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import LdaMulticore
from gensim.corpora import Dictionary

import sys
sys.path.append('../helpers_python')
from pre_processing import *
from helpers import *

def load_data_lda(country):
    df = pd.read_csv('../data/to_be_clustered.csv.gz', compression="gzip")
    df = df[df.whcs == country]
    df.drop(df[df.clean.isna()].index,inplace =True)
    return df

def add_bi_tri_grams(data_words):
    bigram = gensim.models.Phrases(data_words, min_count=3, threshold=1) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=1)  
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    data_bi_tri = [bigram_mod[doc] for doc in data_words]
    data_bi_tri = [trigram_mod[bigram_mod[doc]] for doc in data_words]
    return data_bi_tri
def show_topics(model, nb_word_per_model = 8):
    for topic in model.show_topics(num_words=nb_word_per_model):
        print(topic)

def tune_params(params,corpus,dictionary, nb_topics = None, alphas = None, betas = None): 
    if nb_topics == None :
        nb_topics = [params['LDA']['nb_topics']]
    if alphas == None :
        alphas = [params['LDA']['alpha']]
    if betas == None :
        betas = [params['LDA']['beta']]

    base_beta = params['LDA']['beta']
    base_alpha = params['LDA']['alpha']
    base_k = params['LDA']['nb_topics']
    for k in nb_topics:
        params['LDA']['nb_topics'] = k
        for alpha in alphas :
            params['LDA']['alpha'] = alpha
            for beta in betas:
                params['LDA']['beta'] = beta
                model = get_model(params,corpus, dictionary)
                print("alpha = %3.4f, beta = %3.4f, nb_topics = %d, u_mass %.5f"%(alpha, beta, k, CoherenceModel(model=model, corpus=corpus, coherence='u_mass').get_coherence()))
                show_topics(model)
                print()
    params['LDA']['beta'] = base_beta
    params['LDA']['alpha'] = base_alpha
    params['LDA']['nb_topics'] = base_k
    
def get_model(params, corpus, dictionary):
    return LdaMulticore(corpus=corpus, num_topics=params['LDA']['nb_topics'], alpha = params['LDA']['alpha'], eta = params['LDA']['beta'],
                        id2word=dictionary, workers=6, passes=params['LDA']['passes'], random_state=params['LDA']['random_state'])


def get_dictionary(data, params, more_stop_words = []):
    data_words = data.apply(lambda x : remove_stop_words(str(x), more_stop_words).split(' ')).tolist()
    data_words = add_bi_tri_grams(data_words)
    dictionary = Dictionary(data_words)
    dictionary.filter_extremes(no_below=params['vec_repr']['min_df'], no_above=params['vec_repr']['min_df'])
    return dictionary

def get_corpus_in_bow(data, dictionary, more_stop_words = []):
    data_words = data.apply(lambda x : remove_stop_words(str(x), more_stop_words).split(' ')).tolist()
    data_words = add_bi_tri_grams(data_words)
    corpus = [dictionary.doc2bow(doc) for doc in data_words]
    return corpus