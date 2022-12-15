# Censorship of Twitter - Unsupervised Topic Modeling

This project focuses on topic modeling for censored tweets. In recent years, there has been an increasing amount of censorship on social media platforms such as Twitter. This has led to a need for methods to identify and understand the topics of censored tweets. In our case, we aim to discover topics that are prohibited by some countries.

## Approach

The project uses natural language processing (NLP) techniques to perform topic modeling on censored tweets. This involves a careful pre-processing of the text of the tweets and the use of different topic modeling algorithms. The algorithms used are the following: Latent Dirichlet Allocation (LDA), Biterm Topic Model (BTM), Gibbs Sampling Dirichlet Mixture Model (GSDMM) and BERTopic that uses sentence transformers.

Once the topics have been identified, we analyze and interpret the results to gain insights into the content of the censored tweets.


## Report

The report can be found [here](https://github.com/CS-433/cs-433-project-2-todo.pdf).

Don't forget to add the link!

## Requirements
To run this project, you will need Python 3 and the following libraries:

* bertopic
* bitermplus
* contractions
* deep_translator
* dict_replacement
* gensim
* gsdmm
* hdbscan
* json
* langcodes
* matplotlib
* nltk
* numpy
* pandas
* pickle
* pycountry
* pyLDAvis
* re
* seaborn
* sentence_transformers
* sklearn
* swifter
* textblob
* tmplot
* umap
* wordcloud

You can download the necessary package using the following command:

```pip install -r requirements.txt```


 
## Code organization

The code is split into two main parts:
blablabla

```
.
├── Algorithms
│   ├── BERT-France.ipynb
│   ├── BERT-India.ipynb
│   ├── Bert.ipynb
│   ├── GSDMM.ipynb
│   └── LDA.ipynb
├── BERT-France.ipynb
├── EDA.ipynb
├── README.md
├── __pycache__
│   ├── dict_replacement.cpython-38.pyc
│   ├── helpers.cpython-38.pyc
│   ├── mama.cpython-38.pyc
│   ├── pre_processing.cpython-38.pyc
│   └── preprocessing.cpython-38.pyc
├── censored_tweets
├── data
│   ├── labelling
│   │   ├── France_final.csv.gz
│   │   └── France_labeled.xlsx
│   ├── out_clean.csv.gz
│   ├── saved_models
│   │   └── save_models_for_France.pkl
│   └── to_be_clustered.csv.gz
├── helpers_notebooks
│   ├── Eva_notebook.ipynb
│   ├── Labelling.ipynb
│   ├── Robin_notebook.ipynb
│   ├── Transform DataFrames.ipynb
│   ├── processing_after_traduction.ipynb
│   └── processing_before_traduction.ipynb
├── helpers_python
│   ├── dict_replacement.py
│   ├── helpers.py
│   └── pre_processing.py
├── out_clean.csv
└── requierments.txt
```

## Results

The results of the topic modeling can be used to gain insights into the content of censored tweets. For example, the identified topics can be used to understand what types of content are being censored, and the top words for each topic can provide further information on the specific details of the censored tweets.

Our BERTopic best BERTopic parameters for in the following folder:

## Authors

- Eva Luvison eva.luvison@epfl.ch
- Mathieu Desponds mathieu.desponds@epfl.ch
- Robin Jaccard robin.jaccard@epfl.ch