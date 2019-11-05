import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# Reads data 
genres = pd.read_json('genres.json.gz', orient= 'record', lines = True, encoding='utf-8')
wikidata = pd.read_json('wikidata-movies.json.gz', orient = 'record', lines = True, encoding='utf-8')
omdb =  pd.read_json('omdb-data.json.gz', orient = 'record', lines = True, encoding='utf-8')
rotten_t = pd.read_json('rotten-tomatoes.json.gz', orient = 'record', lines = True, encoding='utf-8')


# changes rating into a label bsed on the rating score 
def rating_to_label(rating):
    if(rating < 4.00):
        return 'Least liked genre'
    
    elif( rating >=4.00 and rating < 5.00):
        return 'Not worth Watching'
    
    elif( rating >=5.00 and rating < 6.00):
        return 'Average'
    
    elif(rating >= 6.00 and rating < 7.00 ):
        return 'Good' 
    
    elif(rating >= 7.00 and rating < 8.00 ):
        return 'Popular' 
    
    elif( rating >= 8.00):
        return 'Excellent'

#Filter any titles that are null from wikidate
wikidata = wikidata[~(wikidata['enwiki_title'].isnull())]

# filters data bzsed on number of audience ratings, NAN valuess
rotten_t = rotten_t[rotten_t['audience_ratings'] > 1000 ].reset_index() 
rotten_t = rotten_t[ ~(rotten_t['critic_average'].isnull()) ]

# merges wikidata and rotten tomatoes
rotten_t = rotten_t.merge(wikidata, on = 'rotten_tomatoes_id')

# takes particular columns from the dataframe
movies_data = rotten_t[['audience_average', 'audience_ratings', 'critic_average', 'cast_member', 'genre', 'label',  'publication_date']	]

# scales audience average to match same scale of critics average
movies_data['audience_average'] = movies_data['audience_average'] * 2

# takes first element of genre and cast members from corresponding column
movies_data['genre'] = movies_data.genre.apply(lambda x: x[0])
movies_data ['cast_member'] = movies_data['cast_member'].str[0]

# changes integer rating to corresponding label
movies_data['audience_average'] = movies_data['audience_average'].apply(lambda x : rating_to_label(x))

movies_data = movies_data[~(movies_data['audience_average'].isnull())]


# label encoder to convert each genre string to a unique number
le = preprocessing.LabelEncoder()

X = movies_data['genre']
X = le.fit_transform(X)
X = X.reshape(-1,1)

y = movies_data['audience_average']
y = y[~(y.isnull())]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25)

model = make_pipeline(
    KNeighborsClassifier(n_neighbors=5)
)
model.fit(X_train, y_train)
predicted = model.predict(X_valid)

X_valid.reshape(3192)

y_valid = y_valid.reset_index()
X_valid = le.inverse_transform(X_valid)

# creates new dataframe to have genre and corresponding true and predicted value
genre_comparison = pd.DataFrame(X_valid, columns = ['Genre'])
genre_comparison.insert(1, 'True_audience_average', y_valid['audience_average']) 
genre_comparison['Predicted_audience_average'] = predicted
genre_comparison['Genre'] = genre_comparison.Genre.replace(genres.set_index('wikidata_id')['genre_label'])


print(genre_comparison[genre_comparison['True_audience_average']== 'Excellent'])

#print(model.score(X_valid, y_valid))