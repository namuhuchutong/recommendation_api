import pickle
import json
from flask import Flask, request
from flask_cors import CORS
from flask_restful import Resource, Api, reqparse
from model import ContentsBased, Hybrid


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


api.add_resource(MovieBase, '/movies/<string:title>')
api.add_resource(UserBase, '/userId/<string:title>')

if __name__ == '__main__':
    app.run(port=5000, debug=True)
