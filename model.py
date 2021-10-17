import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentsBased(object):

    def __init__(self, data):
        self.smd = data
        self.count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
        self.count_matrix = self.count.fit_transform(self.smd['soup'])
        self.cosine_sim = cosine_similarity(self.count_matrix, self.count_matrix)
        self.smd = self.smd.reset_index()
        self.titles = self.smd['title']
        self.indices = pd.Series(self.smd.index, index=self.smd['title'])

    def calc_sim(self, title):
        idx = self.indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:30]
        return sim_scores

    def predict(self, title):
        sim_scores = self.calc_sim(title)
        movie_indices = [i[0] for i in sim_scores]
        return self.titles.iloc[movie_indices]


class Hybrid(object):

    def __init__(self, data, model, indices, contentbase):
        self.smd = data
        self.svd = model
        self.contentbase = contentbase
        self.indices_map = indices

    def hybrid(self, user, title):
        sim_scores = self.contentbase.calc_sim(title)
        movie_indices = [i[0] for i in sim_scores]

        movies = self.smd.iloc[movie_indices][['title', 'id']]
        movies['est'] = movies['id'].apply(lambda x: self.svd.predict(user, self.indices_map.loc[x]['movieId']).est)
        movies = movies.sort_values('est', ascending=False)
        del movies['est']
        return movies.head(20)

    def content_predict(self, title):
        return self.contentbase.predict(title)