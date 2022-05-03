"""

    Single Value Decomposition plus plus (SVDpp) model training.

    Author: Explore Data Science Academy.

    Description: Simple script to train and save an instance of the
    SVDpp algorithm on MovieLens data.

"""
# Script dependencies
import numpy as np
import pandas as pd
from surprise import SVD
import surprise
import pickle
from sklearn.model_selection import train_test_split

# Importing datasets
ratings = pd.read_csv('unsupervised_data./unsupervised_movie_data/train.csv')
ratings.drop('timestamp',axis=1,inplace=True)
trainset, testset = train_test_split(ratings, test_size =0.2, random_state=28)

def svd_pp(save_path):
    # Check the range of the rating
    min_rat = ratings['rating'].min()
    max_rat = ratings['rating'].max()
    # Changing ratings to their standard form
    reader = surprise.Reader(rating_scale = (min_rat,max_rat))
    # Loading the data frame using surprice
    data_load = surprise.Dataset.load_from_df(trainset, reader)
    # Insatntiating surpricce
    method = SVD(lr_all = 0.006 , reg_all = 0.04 , n_epochs = 50 , init_std_dev = 0.05)
    # Loading a trainset into the model
    model = method.fit(data_load.build_full_trainset())
    print (f"Training completed. Saving model to: {save_path}")

    return pickle.dump(model, open(save_path,'wb'))

if __name__ == '__main__':
    svd_pp('team2_SVD.pkl')
