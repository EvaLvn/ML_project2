def remove_hashtags(tweets):
    return re.sub(r"#\w+", " ", tweets)

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
    clean_tweets = clean_tweets.apply(lambda x: remove_hashtags(x))
    clean_tweets = clean_tweets.apply(lambda x: remove_user_mentions(x))
    clean_tweets = clean_tweets.apply(lambda x: remove_numbers(x))
    clean_tweets = clean_tweets.apply(lambda x: remove_RT(x))
    
    return clean_tweets


