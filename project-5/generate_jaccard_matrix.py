###############
### IMPORTS ###
###############

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import nltk

df_all = pd.read_csv('data/dataframe_merged.csv')
df = pd.read_csv('data/dataframe_merged.csv', usecols=['id', 'title', 'genres', 'cast', 'director'])

# This will join first and last names to a single string (and lowercase) so that they do not
# become split during the vectorization process
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

# Apply clean_data function to your features.
features = ['cast', 'director']

for feature in features:
    df[feature] = df[feature].apply(clean_data)

# Clean the text further by only keeping alphanumerics
import re

def create_metasoup(x):
    string = ''.join(x['cast']) + ' ' + x['director']
    return re.sub(r'\W+', ' ', string)
df['metasoup'] = df.apply(create_metasoup, axis=1)

# tokenize dataset
df['split_metasoup'] = df['metasoup'].apply(lambda x: set(nltk.ngrams(nltk.word_tokenize(x), n=1)))

# Convert df['split_metasoup'] into a numpy array for faster computation;
# working in pandas will take ~40 hours
split_metasoup_array = df['split_metasoup'].to_numpy()

# Write a function to calculte jaccard distance
# Without the try-except loop, an error occurs: 'cannot divide by 0', when creating df_jaccard below
# https://python.gotrained.com/nltk-edit-distance-jaccard-distance/#Jaccard_Distance
def calculate_jaccard_dist(metasoup_A, metasoup_B):
    try:
        jaccard = 1 - nltk.jaccard_distance(metasoup_A, metasoup_B)
        return jaccard
    except:
        return 0

# Initialize a blank array that will be filled
jaccard = np.zeros((len(split_metasoup_array), len(split_metasoup_array)))

# This for loop will create the filled array, which is the jaccard_similarity array
for idx in range(0, len(split_metasoup_array)):
    print(idx)
    for idx2 in range(0, len(split_metasoup_array)):
        jaccard[idx, idx2] = calculate_jaccard_dist(split_metasoup_array[idx], split_metasoup_array[idx2])

# Save the numpy array as 'jaccard_metadata.npy' for use in the application
np.save('cosine_similarity/jaccard_metadata.npy', jaccard)
