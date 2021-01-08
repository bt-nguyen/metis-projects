###############
### IMPORTS ###
###############

import pandas as pd
import numpy as np

import matplotlib as plt


import re
import string


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split


import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn import svm
from nltk.tag import StanfordNERTagger
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words
from nltk.tag import pos_tag

# Reference: http://zwmiller.com/projects/nlp_pipeline.html
# Reference: https://github.com/ZWMiller/nlp_pipe_manager/blob/master/nlp_pipeline_manager/nlp_preprocessor.py
# Reference: https://towardsdatascience.com/a-practitioners-guide-to-natural-language-processing-part-i-processing-understanding-text-9f4abfd13e72

# Reference: http://zwmiller.com/projects/nlp_pipeline.html
# Reference: https://github.com/ZWMiller/nlp_pipe_manager/blob/master/nlp_pipeline_manager/nlp_preprocessor.py
# Reference: https://towardsdatascience.com/a-practitioners-guide-to-natural-language-processing-part-i-processing-understanding-text-9f4abfd13e72

# Reference: http://zwmiller.com/projects/nlp_pipeline.html
# Reference: https://github.com/ZWMiller/nlp_pipe_manager/blob/master/nlp_pipeline_manager/nlp_preprocessor.py
# Reference: https://towardsdatascience.com/a-practitioners-guide-to-natural-language-processing-part-i-processing-understanding-text-9f4abfd13e72

class nlp_pipe:

    # Initialize the class
    def __init__(self, vectorizer, stemmer, lemmatizer, tokenizer, dataframe, column='Title'):
        self.vectorizer = vectorizer
        self.tokenizer = tokenizer
        self.lemmatizer = lemmatizer
        self.stemmer = stemmer
        self.dataframe = dataframe
        self.column = column
        self.dataframe[self.column] = self.dataframe[self.column].apply(str)

    ######################################################################

    # Create a cleaning method (aka fit) that will use several functions in order
    def cleaner(self):
        #self.vader_sentiment()
        self.dataframe = self._remove_numbers(self.dataframe, self.column)
        self.dataframe = self._punctuation(self.dataframe, self.column)
        #self.dataframe = self._dropduplicates(self.dataframe, self.column)
        self.real_words() # Check if it's a real word and then remove if not
        self.remove_single_letter() # Remove single letter words
        self.tokenize_words()
        #self.lemmatize_words()
        #self.stem_words()
        self.dataframe = self._join_words(self.dataframe, self.column)
        #self.dataframe[self.column] = self.dataframe[self.column].replace('', np.nan,)
        #self.dataframe.dropna(subset=[self.column], inplace=True)

    ########## Functions that 'cleaner' will call ##########
    @staticmethod
    def _remove_numbers(dataframe, column):
        # Removes all words containing numbers
        remove_numbers = lambda x: re.sub('\w*\d\w*', '', x)
        dataframe[column] = dataframe[column].map(remove_numbers)
        return dataframe

    @staticmethod
    def _punctuation(dataframe, column):
        # Removes punctuation marks
        punc_lower = lambda x: re.sub('[^A-Za-z0-9]+', ' ', x)
        dataframe[column] = dataframe[column].map(punc_lower)
        return dataframe

    @staticmethod
    def _dropduplicates(dataframe, column):
        # Drop rows that have duplicate 'Titles'
        dataframe.drop_duplicates(subset=column, keep='first', inplace=True)
        return dataframe

    @staticmethod
    def _join_words(dataframe, column):
        # Joins words together with space (' ')--used after tokenization
        join_words = lambda x: ' '.join(x)
        dataframe[column] = dataframe[column].map(join_words)
        return dataframe

    def tokenize_words(self):
        self.dataframe[self.column] = self.dataframe.apply(lambda x: self.tokenizer(x[self.column]), axis=1)

    def stem_words(self):
        self.dataframe[self.column] = self.dataframe.apply(lambda x: [self.stemmer.stem(word) for word in x[self.column]], axis=1)

    def lemmatize_words(self):
        self.dataframe[self.column] = self.dataframe.apply(lambda x: [self.lemmatizer.lemmatize(word) for word in x[self.column]], axis=1)

    def real_words(self):
        # Removes words that are not within the nltk.corpus library
        words = set(nltk.corpus.words.words())
        self.dataframe[self.column] = self.dataframe.apply(lambda x: \
        " ".join(w for w in nltk.wordpunct_tokenize(x[self.column]) if w.lower() in words or not w.isalpha()), axis=1)

    def remove_single_letter(self):
        # Removes words that are 1 letter
        self.dataframe[self.column] = self.dataframe.apply(lambda x: ' '.join([w for w in x[self.column].split() if len(w)>2]), axis=1)


df = pd.read_csv('data/dataframe_merged_small.csv', usecols=['id', 'title', 'overview', 'tagline'])

# Replace NaN with empty strings
df['overview'] = df['overview'].replace(np.nan, '', regex=True)
df['tagline'] = df['tagline'].replace(np.nan, '', regex=True)

# Join [overview] and [keywords] together
# These two columns are synopsis-associated and it's sensible to join them together
df['overview_and_tagline'] = df['overview'] + df['tagline']

# Clean the text using nlp_pipelines class
nlp = nlp_pipe(dataframe = df,
               column = 'overview_and_tagline',
               tokenizer = nltk.word_tokenize,
               vectorizer = TfidfVectorizer(stop_words='english'),
               stemmer = SnowballStemmer("english"),
               lemmatizer = WordNetLemmatizer())

nlp.cleaner()

df['overview_and_tagline'] = df['overview_and_tagline'].replace('', 'placeholder', regex=True)

df['tokenize_overview_and_tagline'] = df['overview_and_tagline'].apply(lambda x: x.lower())
df['tokenize_overview_and_tagline'] = df['tokenize_overview_and_tagline'].apply(lambda x: x.split())


df = df.reset_index()

# Reference: https://kanoki.org/2019/03/07/sentence-similarity-in-python-using-doc2vec/
tagged = [TaggedDocument(words=word_tokenize(_d.lower()),
tags = [str(i)]) for i, _d in enumerate(df['overview_and_tagline'])]

model = Doc2Vec.load('data/doc2vec_small.model')

cos_matrix = np.ones((len(df), len(df)))

from itertools import permutations

array_idx = np.arange(0, len(df))

for idx in permutations(array_idx, 2):
    cos_matrix[idx] = model.n_similarity(df['tokenize_overview_and_tagline'][idx[0]], df['tokenize_overview_and_tagline'][idx[1]])
    print(idx)

# # Save cosine_sim array to use in hybrid recommendation system
np.save('similarity_matrix/cos_overview_doc2vec_small.npy', cos_matrix)
