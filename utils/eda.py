

def get_movie_names(data, genre_name):
    
    # set to store movie names
    set_ = set()
    # iterate inside the rows and columns of the movies data
    data = data[['title','genres']]
    for ind, col in data.iterrows():
       # check if the movie has that genre in its genre list
        if genre_name in col['genres'].split():
            # append the movie name to the list
            set_.update({col['title']})

    return list(set_)
# show movies that have no genre associated with them


# chek for the number of unique genres in the movie
# container to store genres( we use set so no one genre would be counted twice)
def unique_genre(x):
    genre_store = set()
    for i in range(x.shape[0]):
        # iterate inside of all the genres in the data
        for genres in x.genres.values[i].split():
            # add each genre to the set
            genre_store.update({genres})
    return list(genre_store)



