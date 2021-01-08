import pandas as pd
import numpy as np

from surprise import accuracy
from surprise import SVD, NMF
from surprise import Dataset
from surprise import Reader
from surprise import NormalPredictor
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise.model_selection import GridSearchCV

# Read in the ratings_small (user-item) dataframe
df_ratings = pd.read_csv('data/ratings.csv')

df_all = pd.read_csv('data/dataframe_merged.csv', usecols=['id', 'title'])

df_ratings = df_ratings[df_ratings['movieId'].isin(df_all['id'])]

# Set the reader to have a rating_scale from 1-5 (default)
reader = Reader(rating_scale=(1, 5))

# The data only consists of userId, movieId, and rating
data = Dataset.load_from_df(df_ratings[['userId', 'movieId', 'rating']], reader)

# Use surprise package for a train-test split of 80-20
# Note that the train-test split will split by general rows, not specific users
from surprise.model_selection import train_test_split
trainset, testset = train_test_split(data, test_size=0.20)

trainset_iids = list(trainset.all_items())
iid_converter = lambda x: trainset.to_raw_iid(x)
trainset_raw_iids = list(map(iid_converter, trainset_iids))

from surprise import KNNWithMeans
my_k = 15
my_min_k = 5
my_sim_option = {
    'name':'cosine', 'user_based':False, 'verbose': False
    }
algo = KNNWithMeans(sim_options = my_sim_option)
algo.fit(trainset)

# Same dataframe as algo.sim but the indices/columns are now movieId
df_cos_surprise = pd.DataFrame(algo.sim, index=trainset_raw_iids, columns=trainset_raw_iids)

df_all = df_all.reset_index()
df_all.index = df_all.id

movieIdtoindex = df_all['index'].to_dict()

df_cos_surprise = df_cos_surprise.rename(index=movieIdtoindex, columns=movieIdtoindex)

# Make a pandas dataframe of movie x movie length from df_all
# Fill in the values from matrix 'algo.sim'
# Set the diagonal to "1"
df_blank = pd.DataFrame(np.nan, range(1,len(df_all)), range(1,len(df_all)))

df_blank = df_cos_surprise.combine_first(df_blank)

np.fill_diagonal(df_blank.values, 1)
df_bank = df_blank.fillna(0)

np.save('cosine_similarity/cos_ratings_all.npy', df_blank)
