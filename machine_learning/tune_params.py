import sys

sys.path.append('..')

from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

import pymongo

from utils.utils import types
from utils.utils import get_data


client = pymongo.MongoClient('mongodb://localhost:27017/')

db = client['news_db']

train_collection = db['training_news']
test_collection = db['testing_news']


model_name = sys.argv[1]


size = 1250
train_size = int(0.8 * size)
test_size = int(0.2 * size)


train_content, train_types = get_data(train_collection, train_size)
test_content, test_types = get_data(test_collection, test_size)


tfidf = TfidfVectorizer(
    min_df=2,
    max_df=0.5,
    ngram_range=(1, 2),
    sublinear_tf=True,
    norm='l2'
)

params_tfidf = {
    'tfidf__min_df': [1, 2, 5, 10],
    'tfidf__max_df': [0.5, 0.6, 0.7],
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidf__sublinear_tf': [True, False],
    'tfidf__norm': ['l1', 'l2']
}


pipeline_SGD = Pipeline([
    ('tfidf', tfidf),
    ('model', OneVsRestClassifier(SGDClassifier()))
])

params_SGD = {
    'model__estimator__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
    'model__estimator__penalty': ['l2', 'l1', 'elasticnet'],
    'model__estimator__alpha': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
}


pipeline_SVC = Pipeline([
    ('tfidf', tfidf),
    ('model', OneVsRestClassifier(LinearSVC()))
])

params_SVC = {
    'model__estimator__penalty': ['l1', 'l2'],
    'model__estimator__loss': ['hinge', 'squared_hinge'],
    'model__estimator__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}


pipeline_NB = Pipeline([
    ('tfidf', tfidf),
    ('model', OneVsRestClassifier(MultinomialNB()))
])

params_NB = {
    'model__estimator__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'model__estimator__fit_prior': [True, False]
}


pipeline_PAC = Pipeline([
    ('tfidf', tfidf),
    ('model', OneVsRestClassifier(PassiveAggressiveClassifier()))
])

params_PAC = {
    'model__estimator__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'model__estimator__loss': ['hinge', 'squared_hinge']
}


if model_name == 'SGD':
    pipeline = pipeline_SGD
    params = params_SGD
elif model_name == 'SVC':
    pipeline = pipeline_SVC
    params = params_SVC
elif model_name == 'NB':
    pipeline = pipeline_NB
    params = params_NB
elif model_name == 'PAC':
    pipeline = pipeline_PAC
    params = params_PAC


grid_search_tune = GridSearchCV(pipeline, params, cv=2)
grid_search_tune.fit(train_content, train_types)

print(f'{model_name} best parameters:')
print(grid_search_tune.best_params_)

print(f'Best cross-validation accuracy: {round(grid_search_tune.best_score_ * 100, 2)}%')

pred = grid_search_tune.predict(test_content)
print(classification_report(test_types, pred, target_names=types))

acc = accuracy_score(test_types, pred)
print(f'Accuracy: {round(acc * 100, 2)}%')
