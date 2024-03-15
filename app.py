from flask import Flask, render_template, request

import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.kernel_approximation import svd
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

app = Flask(__name__)

credits = pd.read_csv("archive/credits.csv")
keywords = pd.read_csv("archive/keywords.csv")
links_small = pd.read_csv("archive/links_small.csv")
md = pd.read_csv("archive/movies_metadata.csv", low_memory=False)
ratings = pd.read_csv("archive/ratings_small.csv")


md["genres"] = (
    md["genres"]
    .fillna("[]")
    .apply(literal_eval)
    .apply(lambda x: [i["name"] for i in x] if isinstance(x, list) else [])
)

# this is V  -> popularity
vote_counts = md[md["vote_count"].notnull()]["vote_count"].astype("int")

# this is R  -> average rating
vote_averages = md[md["vote_average"].notnull()]["vote_average"].astype("int")

# this is C -> mean of vote average.
C = vote_averages.mean()
m = vote_counts.quantile(0.95)

md["year"] = pd.to_datetime(md["release_date"], errors="coerce").apply(
    lambda x: str(x).split("-")[0] if x != np.nan else np.nan
)

# filters movies based on vote count, not null values.
qualified = md[
    (md["vote_count"] >= m)
    & (md["vote_count"].notnull())
    & (md["vote_average"].notnull())
][["title", "year", "vote_count", "vote_average", "popularity", "genres"]]

qualified["vote_count"] = qualified["vote_count"].astype("int")
qualified["vote_average"] = qualified["vote_average"].astype("int")


# calculate rating based on its own rating from different users.
def weighted_rating(x):
    v = x["vote_count"]
    R = x["vote_average"]
    return (v / (v + m) * R) + (m / (m + v) * C)


qualified["wr"] = qualified.apply(weighted_rating, axis=1)
qualified = qualified.sort_values("wr", ascending=False).head(250)

# the genre is expanded into multiple columns.
s = (
    md.apply(lambda x: pd.Series(x["genres"]), axis=1)
    .stack()
    .reset_index(level=1, drop=True)
)
s.name = "genre"
gen_md = md.drop("genres", axis=1).join(s)
gen_md.head(3).transpose()


# percentile -> compares individual data points to the overall distribution.
def build_chart(genre, percentile=0.85):
    # filter movie based on genre
    df = gen_md[gen_md["genre"] == genre]

    # extract vote count and average
    vote_counts = df[df["vote_count"].notnull()]["vote_count"].astype("int")
    vote_averages = df[df["vote_average"].notnull()]["vote_average"].astype("int")
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    # select movies with vote counts greater than equal to m

    qualified = df[
        (df["vote_count"] >= m)
        & (df["vote_count"].notnull())
        & (df["vote_average"].notnull())
    ][["title", "year", "vote_count", "vote_average", "popularity"]]
    qualified["vote_count"] = qualified["vote_count"].astype("int")
    qualified["vote_average"] = qualified["vote_average"].astype("int")
    # calculate wr for qualified movie

    qualified["wr"] = qualified.apply(
        lambda x: (x["vote_count"] / (x["vote_count"] + m) * x["vote_average"])
        + (m / (m + x["vote_count"]) * C),
        axis=1,
    )
    # sorting and selecting top movies
    qualified = qualified.sort_values("wr", ascending=False).head(250)

    return qualified


# Content based recommendation system
links_small = links_small[links_small["tmdbId"].notnull()]["tmdbId"].astype("int")

## Pre-processing step


def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan


# convert id into integer using convert_int function
md["id"] = md["id"].apply(convert_int)
# identify id is null.
# md[md['id'].isnull()]
md = md.drop([19730, 29503, 35587])

smd = md[md["id"].isin(links_small)]
smd.loc[:, "tagline"] = smd["tagline"].fillna("")
smd.loc[:, "description"] = smd["overview"] + smd["tagline"]
smd.loc[:, "description"] = smd["description"].fillna("")


tfidf_vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
tfidf_matrix = tfidf_vectorizer.fit_transform(smd["description"])


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

smd = smd.reset_index()
titles = smd["title"]
indices = pd.Series(smd.index, index=smd["title"])


def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


# Content based RS : Using movie description, taglines, keywords, cast, director and genres

keywords["id"] = keywords["id"].astype("int")
credits["id"] = credits["id"].astype("int")
md["id"] = md["id"].astype("int")

md = md.merge(credits, on="id")
md = md.merge(keywords, on="id")
smd = md[md["id"].isin(links_small)]

smd["cast"] = smd["cast"].apply(literal_eval)
smd.loc[:, "crew"] = smd["crew"].apply(literal_eval)
smd.loc[:, "keywords"] = smd["keywords"].apply(literal_eval)
smd.loc[:, "cast_size"] = smd["cast"].apply(lambda x: len(x))
smd.loc[:, "crew_size"] = smd["crew"].apply(lambda x: len(x))


def get_director(x):
    for i in x:
        if i["job"] == "Director":
            return i["name"]
    return np.nan


smd.loc[:, "director"] = smd["crew"].apply(get_director)
smd.loc[:, "cast"] = smd["cast"].apply(
    lambda x: [i["name"] for i in x] if isinstance(x, list) else []
)
smd.loc[:, "cast"] = smd["cast"].apply(lambda x: x[:3] if len(x) >= 3 else x)
smd.loc[:, "keywords"] = smd["keywords"].apply(
    lambda x: [i["name"] for i in x] if isinstance(x, list) else []
)

smd.loc[:, "cast"] = smd["cast"].apply(
    lambda x: [str.lower(i.replace(" ", "")) for i in x]
)
smd.loc[:, "director"] = (
    smd["director"].astype("str").apply(lambda x: str.lower(x.replace(" ", "")))
)
smd.loc[:, "director"] = smd["director"].apply(lambda x: [x, x, x])


s = (
    smd.apply(lambda x: pd.Series(x["keywords"]), axis=1)
    .stack()
    .reset_index(level=1, drop=True)
)
s.name = "keyword"
s = s.value_counts()

s = s[s > 1]

stemmer = SnowballStemmer("english")


def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words


smd.loc[:, "keywords"] = smd["keywords"].apply(filter_keywords)
smd.loc[:, "keywords"] = smd["keywords"].apply(lambda x: [stemmer.stem(i) for i in x])
smd.loc[:, "keywords"] = smd["keywords"].apply(
    lambda x: [str.lower(i.replace(" ", "")) for i in x]
)

smd.loc[:, "soup"] = smd["keywords"] + smd["cast"] + smd["director"] + smd["genres"]
smd.loc[:, "soup"] = smd["soup"].apply(lambda x: " ".join(x))

# matrix represents the count or frequency of a token
count_vectorizer = CountVectorizer(stop_words="english", min_df=1)
count_matrix = count_vectorizer.fit_transform(smd["soup"])


cosine_sim = cosine_similarity(count_matrix, count_matrix)

smd = smd.reset_index()
titles = smd["title"]
indices = pd.Series(smd.index, index=smd["title"])


def improved_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]

    movies = smd.iloc[movie_indices][["title", "vote_count", "vote_average", "year"]]
    vote_counts = movies[movies["vote_count"].notnull()]["vote_count"].astype("int")
    vote_averages = movies[movies["vote_average"].notnull()]["vote_average"].astype(
        "int"
    )
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[
        (movies["vote_count"] >= m)
        & (movies["vote_count"].notnull())
        & (movies["vote_average"].notnull())
    ]
    qualified["vote_count"] = qualified["vote_count"].astype("int")
    qualified["vote_average"] = qualified["vote_average"].astype("int")
    qualified["wr"] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values("wr", ascending=False).head(10)
    return qualified


# CF based recommendation system
from surprise import SVD
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split, cross_validate
import requests

# Assuming you have already defined 'reader' and 'ratings'
reader = Reader()
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Create an instance of the SVD model
svd = SVD()

# Use cross_validate for cross-validation
cross_validate(svd, data, measures=["RMSE", "MAE"], cv=5, verbose=True)

# Fit the model to the entire training set
trainset = data.build_full_trainset()
svd.fit(trainset)

# Example: Predict rating for user 1 and item 302
predicted_rating = svd.predict(1, 302).est
# print(f"Predicted rating for user 1 and item 302: {predicted_rating}")


# Hybrid recommendation system
def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan


id_map = pd.read_csv("./archive/links_small.csv")[["movieId", "tmdbId"]]
id_map["tmdbId"] = id_map["tmdbId"].apply(convert_int)
id_map.columns = ["movieId", "id"]
id_map = id_map.merge(smd[["title", "id"]], on="id").set_index("title")
# id_map = id_map.set_index('tmdbId')

indices_map = id_map.set_index("id")


def hybrid(title, userId=123):
    try:
        idx = indices[title]
        tmdbId = id_map.loc[title]["id"]
        movie_id = id_map.loc[title]["movieId"]
        sim_scores = list(enumerate(cosine_sim[int(idx)]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:26]
        movie_indices = [i[0] for i in sim_scores]
        movies = smd.iloc[movie_indices][
            ["title", "vote_count", "vote_average", "release_date", "id"]
        ]
        movies["est"] = movies["id"].apply(
            lambda x: svd.predict(userId, indices_map.loc[x]["movieId"]).est
        )
        movies = movies.sort_values("est", ascending=False)

        # Return movie details along with names
        return movies.head(12)
    except KeyError:
        raise IndexError("Movie not found. Please enter a valid movie title.")


# Replace with your TMDB API key
TMDB_API_KEY = "e65f96397db5471ad7bab643b6f327ca"


def get_movie_details(movie_title):
    base_url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": movie_title,
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if "results" in data and data["results"]:
        movie_details = data["results"][0]
        return {
            "title": movie_details.get("title", ""),
            "overview": movie_details.get("overview", ""),
            "poster_path": "https://image.tmdb.org/t/p/w500/"
            + movie_details.get("poster_path", ""),
        }

    raise IndexError(f"Movie not found: {movie_title}")


def get_recommendations(movie_title):
    search_url = "https://api.themoviedb.org/3/search/movie"
    recommendations_url = (
        "https://api.themoviedb.org/3/movie/{movie_id}/recommendations"
    )

    # Step 1: Search for the given movie title
    search_params = {
        "api_key": TMDB_API_KEY,
        "query": movie_title,
    }

    search_response = requests.get(search_url, params=search_params)
    search_data = search_response.json()

    if "results" in search_data and search_data["results"]:
        movie_id = search_data["results"][0].get("id")

        # Step 2: Get recommendations for the found movie ID
        recommendations_params = {
            "api_key": TMDB_API_KEY,
        }

        recommendations_response = requests.get(
            recommendations_url.format(movie_id=movie_id), params=recommendations_params
        )
        recommendations_data = recommendations_response.json()

        recommendations = []
        for result in recommendations_data.get("results", [])[:13]:
            recommendations.append(
                {
                    "title": result.get("title", ""),
                    "overview": result.get("overview", ""),
                    "poster_path": "https://image.tmdb.org/t/p/w500/"
                    + result.get("poster_path", ""),
                }
            )

        return recommendations

    else:
        raise IndexError(f"Movie not found: {movie_title}")


@app.route("/")
def home():
    return render_template("index2.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    if request.method == "POST":
        movie_title = request.form["movie_title"]
        try:
            # Use hybrid function to get recommendations
            recommendations_data = hybrid(movie_title)
            recommendations = []
            for index, row in recommendations_data.iterrows():
                recommendations.append(
                    {
                        "title": row["title"],
                        "overview": get_movie_details(row["title"])["overview"],
                        "poster_path": get_movie_details(row["title"])["poster_path"],
                    }
                )

            return render_template(
                "index2.html",
                movie_details=get_movie_details(movie_title),
                recommendations=recommendations,
            )

        except IndexError as e:
            print(f"Error in getting movie details: {e}")
            return render_template("index2.html", error_message=str(e))


if __name__ == "__main__":
    app.run(debug=True)
