import os
import pandas as pd
import json

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