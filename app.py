import pickle
import json

import pandas as pd
from flask import Flask
from flask_cors import CORS
from flask_restful import Resource, Api, reqparse
from model import ContentsBased, Hybrid

import requests

class Recommender:

    def __init__(self):
        with open('dataset.pickle', 'rb') as f:
            self.dataset = pickle.load(f)
        with open('linkMap.pickle', 'rb') as f:
            self.indices = pickle.load(f)
        with open('svd.pickle', 'rb') as f:
            self.svd = pickle.load(f)

        self.content = ContentsBased(self.dataset)
        self.hybrid = Hybrid(self.dataset, self.svd, self.indices, self.content)

    def parsing_args(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('id', required=False,
                                 type=int,
                                 help="userId")

    def get_content_recommend(self, title):
        if title in self.dataset.values:
            output = {
                'content': list(self.content.predict(title))
            }
        else:
            output = None
        return output

    def get_user_recommend(self, title, user):
        if title in self.dataset.values:
            output = self.hybrid.hybrid(user, title)
            print(output)
        else:
            output = None
        return output

app = Flask(__name__)
api = Api(app)
CORS(app)

recommeder = Recommender()
recommeder.parsing_args()

class MovieBase(Resource):

    def get(self, title):
        title = title.replace('-', ' ')
        output = recommeder.get_content_recommend(title)
        return output


class UserBase(Resource):

    def get(self, title):
        args = recommeder.parser.parse_args()
        userId = args['id']
        output = recommeder.get_user_recommend(title, int(userId))
        return json.loads(output.to_json(orient = 'records'))


class UserInfo(Resource):

    def __init__(self):
        self.movieTitles = []

    def ratingsUser(self, userId):
        ratings = pd.read_csv('dataset/ratings_small.csv')
        user = ratings[ratings['userId'] == userId]

        if len(user) != 0:
            user['movieId'] = user['movieId'].apply(lambda x: self.tmdbToimdb(str(x)))
            user = user[user['movieId'] != 'N/A']
            user['title'] = self.movieTitles
            print(user)
            print(self.movieTitles)
            return user
        else:
            return None

    def tmdbToimdb(self, tmdbId):
        self.movieTitle = []
        apiUrl = 'https://api.themoviedb.org/3/movie/'
        apiKey = 'a0735a2be9600e8356b5d672781cb382'
        URL = apiUrl + tmdbId + '?api_key=' + apiKey
        res = requests.get(URL)

        if res.status_code == 200:
            output = res.json()
            self.movieTitles.append(output['title'])
            output = output['imdb_id']
            return output
        else:
            return 'N/A'

    def get(self):
        args = recommeder.parser.parse_args()
        userId = args['id']
        output = self.ratingsUser(userId)
        return json.loads(output.to_json(orient = 'records'))


api.add_resource(MovieBase, '/api/movies/<string:title>')
api.add_resource(UserBase, '/api/userId/<string:title>')
api.add_resource(UserInfo, '/api/userInfo')

if __name__ == '__main__':
    app.run(port=5000, debug=False)
