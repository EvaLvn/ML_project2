import pickle
import pandas as pd

import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
from umap import UMAP


def save_model(model, params, message = ""):
    saved_models.append((model, params, message))
    
def write_good_params():
    pickle.dump(saved_models, open( "../data/saved_models/save_models_for_"+country+".pkl", "wb" ))
    
def get_model(params, additional_stop_words =  []) :
    embedding_model = SentenceTransformer("all-mpnet-base-v2") #'digio/Twitter4SSE'
    s = list(stopwords.words('english')).extend(additional_stop_words)
    vectorizer_model = CountVectorizer(stop_words=s)
    
    umap_model = UMAP(n_neighbors = params['UMAP']['n_neighbors'], 
                  n_components = params['UMAP']['n_components'], 
                  min_dist = params['UMAP']['min_dist'], 
                  metric = params['UMAP']['metric'], 
                  low_memory = params['UMAP']['low_memory'], 
                  random_state = params['UMAP']['random_state'])

    hdbscan_model = HDBSCAN(min_cluster_size = params['HDBSCAN']['min_cluster_size'],
                       min_samples = params['HDBSCAN']['min_samples'],
                       cluster_selection_epsilon = params['HDBSCAN']['cluster_selection_epsilon'],
                       metric = params['HDBSCAN']['metric'],                      
                       cluster_selection_method = params['HDBSCAN']['cluster_selection_method'],
                       prediction_data = params['HDBSCAN']['prediction_data'])

    model = BERTopic(
        umap_model = umap_model,
        vectorizer_model=vectorizer_model,
        hdbscan_model = hdbscan_model,
        embedding_model=embedding_model,
        language='english', calculate_probabilities=False,
        verbose=True
    )
    return model

def load_data_bert(country):
    df = pd.read_csv('../data/to_be_clustered.csv.gz', compression="gzip")
    df = df[df.whcs == country]
    df.drop(df[df.clean.isna()].index,inplace =True)
    return df

def get_tweets_of_topic(topics, topic_nb, tweets, n):
    return [tweet for i, tweet in enumerate(tweets) if topics[i] == topic_nb][:n]

def get_coherence(model, tweets, topics, coherence = 'c_v'):    # Preprocess Documents
    documents = pd.DataFrame({"Document": tweets,
                              "ID": range(len(tweets)),
                              "Topic": topics})
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    cleaned_docs = model._preprocess_text(documents_per_topic.Document.values)

    # Extract vectorizer and analyzer from BERTopic
    vectorizer = model.vectorizer_model
    analyzer = vectorizer.build_analyzer()

    # Extract features for Topic Coherence evaluation
    words = vectorizer.get_feature_names()
    tokens = [analyzer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topic_words = [[words for words, _ in model.get_topic(topic)] 
                   for topic in range(len(set(topics))-1)]
    # Evaluate
    coherence_model = CoherenceModel(topics=topic_words, 
                                     texts=tokens, 
                                     corpus=corpus,
                                     dictionary=dictionary, 
                                     coherence=coherence)
    coherence = coherence_model.get_coherence()

    return coherence