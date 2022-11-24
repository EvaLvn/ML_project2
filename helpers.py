import os
import pandas as pd
import json
import pycountry

# def create_data():

#     dfs = []
#     for r, d, f in os.walk('./censored_tweets/'):
#         for file in f:
#             print('file')
#             if 'withheldtweets.json' in file: 
#                 print('if')
#                 dfs.append(pd.read_json('./censored_tweets/%s' % file, lines=True))

#     df_cen = pd.concat(dfs)
#     df_cen = df_cen.dropna(subset=['withheld_in_countries'])

#     return df_cen

dfs = []
for r, d, f in os.walk(os.getcwd()):
    for file in f:
        if 'withheldtweets.json' in file:
            dfs.append(pd.read_json("./censored_tweets/%s" % file, lines=True))

df_cen = pd.concat(dfs)
df_cen = df_cen.dropna(subset=['withheld_in_countries'])

def get_name_country(x):
    country = pycountry.countries.get(alpha_2=x)
    if country == None:
        return 'Undefined'
    return country.name

def eda_processing(df):
    df['possibly_sensitive'] = df.possibly_sensitive.apply(lambda x: x == 1 )
    df = df.drop(['contributors', 'geo', 'coordinates', 'quote_count', 'reply_count', 'retweet_count', 'favorite_count', 'favorited', 'retweeted', 'filter_level'], axis = 1)
    return df

def flat_withhelded_countries(df, keep_duplicates = False):
    """
    Take a df as input and look at all the tweets that are withhelded in more than one country
    Then add a tweet for each withhelded country it has in its list 
    Finally add a column if this tweet was duplicated or not.
    
    Ex t1 has withhelded country ['fr', 'ge']. t1 will be added twice in the df once with withhelded country 'fr' and once with withhelded country 'ge'. If keep_duplicates is true, then the original tweet with withhelded country ['fr','ge'] will be kept in df
    """
    
    df_cen_ext = df.copy()
    df_cen_ext['whcs'] = df_cen_ext['whcs'].apply(lambda whc : whc.split(', '))
    df_cen_ext['duplicated'] = df_cen_ext['whcs'].apply(lambda whc : len(whc)>=2)
    df_doubled = df_cen_ext[[len(whcs) >= 2 for whcs in df_cen_ext['whcs']]].copy()
    df_doubled['whcs'] = df_doubled['whcs'].apply(lambda x : ', '.join(x))
    df_cen_ext = df_cen_ext.explode('whcs')
    if keep_duplicates :
        df_cen_ext = pd.concat([df_cen_ext, df_doubled])
        
    return df_cen_ext