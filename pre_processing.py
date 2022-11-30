import re
from dict_replacement import *
from nltk.corpus import stopwords
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

nltk.download
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('punkt')

def remove_user_mentions(tweets):
    return re.sub(r"@\w+", " ", tweets)

def remove_numbers(tweets):
	return re.sub(r"\d+([.,]\d+)?", " ", tweets)

def remove_URLs(tweets):
    return re.sub('((www.[^s]+)|(https?://[^s]+))',' ',tweets)

def remove_RT(tweets):
    return re.sub("RT  :", " ", tweets)

def clean_tweets(df_text):
    clean_tweets = df_text.apply(lambda x: remove_URLs(x))
    clean_tweets = clean_tweets.apply(lambda x: remove_user_mentions(x))
    clean_tweets = clean_tweets.apply(lambda x: remove_numbers(x))
    clean_tweets = clean_tweets.apply(lambda x: remove_RT(x))
    
    return clean_tweets


def clean_tweets_after_trad(df_text):
	clean_tweets = df_text.apply(lambda x: replace_CamelCases(x))
	clean_tweets = clean_tweets.apply(lambda x: x.lower())
	clean_tweets = clean_tweets.apply(lambda x: replace_dict(x))
	clean_tweets = clean_tweets.apply(lambda x: remove_char(x))
	#clean_tweets = pd.DataFrame([ele for ele in clean_tweets if ele != ''], columns =['text'])['text']
	clean_tweets = clean_tweets.apply(lambda x : lemmatize_text(x))
	clean_tweets = clean_tweets.apply(lambda x: remove_stop_words(x))
	return clean_tweets

english_stopwords = stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()

def replace_CamelCases(text):
    return re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', str(text))

def remove_char(tweets):
    return re.sub(r"[^a-zA-Z0-9]", " ", str(tweets))

def remove_stop_words(text):
    tokens = word_tokenize(text.lower())
    tokens_wo_stopwords = [t for t in tokens if t not in english_stopwords]
    return " ".join(tokens_wo_stopwords)

def lemmatize_text(text):
    a = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(text)]
    return ' '.join([lemmatizer.lemmatize(w) for w in a])

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def replace_dict(text):
    for key, value in my_dict.items():
    	text = text.replace(key, value)
    return text