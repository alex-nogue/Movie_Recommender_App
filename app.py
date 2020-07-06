from flask import Flask, render_template, request
import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textdistance as td

app = Flask(__name__)

data_api = pd.read_csv('data_api.csv')

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(data_api['soup'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)

data_api = data_api.reset_index()
indices = pd.Series(data_api.index, index=data_api['movie_title'])
indices = indices.reset_index()


@app.route('/')
def home():
    return render_template('index.html')


#@app.route("/recommend/<string:wrd>")
#def recommend(wrd):
@app.route("/recommend", methods=['POST'])
def recommend():
    wrd = str(request.form.get('word'))
    message = "Displaying recommentations for " + wrd + ":\n"
    if wrd not in indices['movie_title'].unique():
        temp = indices
        temp['movie_name_distance'] = indices.apply(lambda row: td.jaro_winkler(row['movie_title'], wrd), axis=1)
        wrd = temp.sort_values('movie_name_distance', ascending=False).iloc[0,0]
        message = "Couldn't find the input movie, displaying recommendations for " + wrd + ":<br/>"
    idx = int(indices[indices['movie_title'] == wrd][0])
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    recommendations = data_api['movie_title'].iloc[movie_indices].str.cat(sep='<br/>')
    #return message + recommendations
    #return render_template('index.html', prediction='Recommended movies {}'.format(json.dumps(message + recommendations)))
    return render_template('index.html', prediction=message + recommendations)

if __name__ == '__main__':
    app.run(debug=True)
