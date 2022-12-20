import re
from dict_replacement import *
from nltk.corpus import stopwords
import nltk
import contractions
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
    """Removes mentions of user
    Args:
        tweets : the tweets to modifiy
    Returns:
        the modified tweets
    """
    return re.sub(r"@\w+", " ", tweets)

def remove_numbers(tweets):
    """Removes numbers
    Args:
        tweets : the tweets to modifiy
    Returns:
        the modified tweets
    """
    return re.sub(r"\d+([.,]\d+)?", " ", tweets)

def remove_URLs(tweets):   
    """Removes URLs
    Args:
        tweets : the tweets to modifiy
    Returns:
        the modified tweets
    """
    return re.sub(r'http\S+', '', tweets)

def remove_RT(tweets):
    """Removes mentions of retweet
    Args:
        tweets : the tweets to modifiy
    Returns:
        the modified tweets
    """
    return re.sub("RT  :", " ", tweets)

def replace_dict_before_trad(text):
    """Replaces some words by other for uniformity
    Args:
        text : the text to modifiy
    Returns:
        the modified text
    """
    for key, value in my_dict_before_trad.items():
        text = text.replace(key, value)
    return text

def remove_n(tweets):
    """Removes back to line symbol
    Args:
        tweets : the tweets to modifiy
    Returns:
        the modified tweets
    """
    return tweets.replace("\n", " ")

def remove_emojis(data):
    """Removes emojis
    Args:
        data : the data to modifiy
    Returns:
        the modified data
    """
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)

def clean_tweets(df_text):
    """Cleans tweets
    Args:
        df_text : the texts in dataframe format
    Returns:
        the cleaned texts in dataframe format
    """
    clean_tweets = df_text.apply(lambda x: remove_URLs(x))
    clean_tweets = clean_tweets.apply(lambda x: remove_user_mentions(x))
    clean_tweets = clean_tweets.apply(lambda x: replace_dict_before_trad(x))
    clean_tweets = clean_tweets.apply(lambda x: remove_n(x))
    clean_tweets = clean_tweets.apply(lambda x: remove_emojis(x))
    clean_tweets = clean_tweets.apply(lambda x: remove_numbers(x))
    clean_tweets = clean_tweets.apply(lambda x: remove_RT(x))
    
    return clean_tweets

def clean_tweets_after_trad(df_text):
    """Cleans tweets after the translation
    Args:
        df_text : the texts in dataframe format    
    Returns:
        the cleaned texts in dataframe format
    """
    clean_tweets = df_text.apply(lambda x: replace_CamelCases(x))
    clean_tweets = clean_tweets.apply(lambda x: x.lower())
    clean_tweets = clean_tweets.apply(lambda x: remove_contractions(x))
    clean_tweets = clean_tweets.apply(lambda x: replace_dict(x))
    clean_tweets = clean_tweets.apply(lambda x: remove_char(x))
    clean_tweets = clean_tweets.apply(lambda x: remove_extra_white_space(x))
    #clean_tweets = pd.DataFrame([ele for ele in clean_tweets if ele != ''], columns =['text'])['text']
    #clean_tweets = clean_tweets.apply(lambda x : lemmatize_text(x))
    #clean_tweets = clean_tweets.apply(lambda x: remove_stop_words(x))
    return clean_tweets

english_stopwords = stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()

def replace_CamelCases(text):
    """Replaces camel cases
    Args:
        text : the text to modifiy
    Returns:
        the modified text
    """
    return re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', str(text))

def remove_char(tweets):
    """Removes special characters
    Args:
        tweets : the tweets to modifiy
    Returns:
        the modified tweets
    """
    return re.sub(r"[^a-zA-Z0-9,.!?']", " ", str(tweets))
    #return re.sub(r"[^a-zA-Z0-9]", " ", str(tweets))

def remove_stop_words(text, stop_words = english_stopwords):
    """Removes stop words
    Args:
        text : the text to modifiy
        stop_words : the stop words dictionnary
    Returns:
        the modified text
    """
    tokens = word_tokenize(text.lower())
    tokens_wo_stopwords = [t for t in tokens if t not in stop_words]
    return " ".join(tokens_wo_stopwords)

def lemmatize_text(text):
    """Lemmatizes the text
    Args:
        text : the text to modifiy
    Returns:
        the lemmatized text
    """
    a = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(text)]
    return ' '.join([lemmatizer.lemmatize(w) for w in a])

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts
    Args:
        word : the word to tag
    Returns:
        the corresponding tag
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def replace_dict(text):
    """Replaces some words by other for uniformity
    Args:
        text : the text to modifiy
    Returns:
        the modified text
    """
    for key, value in my_dict.items():
    	text = text.replace(key, value)
    return text

def remove_contractions(text):
    """Removes word contractions
    Args:
        text : the text to modifiy
    Returns:
        the modified text
    """
    # creating an empty list
    expanded_words = []   
    for word in text.split():
    # using contractions.fix to expand the shortened words
        expanded_words.append(contractions.fix(word))  
   
    return ' '.join(expanded_words)

def remove_extra_white_space(text):
    """Removes extra white spaces
    Args:
        text : the text to modifiy
    Returns:
        the modified text
    """
    return re.sub(' +', ' ', text)

