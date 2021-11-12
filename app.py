import pandas as pd
from flask import render_template, request, redirect
from flask import Flask
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return render_template('index.html')


@app.route('/recommender')
def show():  # put application's code here
    return render_template('recommender.html')


df = pd.read_csv('https://raw.githubusercontent.com/codeheroku/Introduction-to-Machine-Learning/master/Building'
                 '%20a%20Movie%20Recommendation%20Engine/movie_dataset.csv')


def combine_features(row):
    return row["keywords"]+" "+row["cast"]+" "+row["genres"]+" "+row["director"]


def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]


def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]


@app.route('/recommend', methods=['GET', 'POST'])
def recommend():  # put application's code here
    movie_user_likes = request.form['movie_user_likes']
    features = ['keywords', 'cast', 'genres', 'director']
    for feature in features:
        df[feature] = df[feature].fillna('')
    df["combined_features"] = df.apply(combine_features, axis=1)
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(df["combined_features"])
    cosine_sim = cosine_similarity(matrix)
    movie_index = get_index_from_title(movie_user_likes)
    similar_movies = list(enumerate(cosine_sim[movie_index]))
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    count = 0
    data = []
    for movie in sorted_similar_movies:
        data.append(get_title_from_index(movie[0]))
        count += 1
        if count >= 10:
            break
    return render_template('recommender.html', data=data)


if __name__ == '__main__':
    app.run()
