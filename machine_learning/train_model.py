import sys

sys.path.append('..')

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier

import joblib
import pymongo

from utils.utils import TRAIN_SIZE
from utils.utils import get_types


client = pymongo.MongoClient('mongodb://localhost:27017/')

db = client['news_db']

train_collection = db['training_news']
test_collection = db['testing_news']


model_name = sys.argv[1]


train_data = joblib.load('models/train_data.pkl')
train_types = get_types(train_collection, TRAIN_SIZE)


if model_name == 'SGD':
    model = OneVsRestClassifier(SGDClassifier(
        loss='hinge',
        penalty='l2',
        alpha=1e-5
    ))
elif model_name == 'SVC':
    model = OneVsRestClassifier(LinearSVC(
        penalty='l2',
        loss='squared_hinge',
        C=1.0
    ))
elif model_name == 'NB':
    model = OneVsRestClassifier(MultinomialNB(
        alpha=0.01,
        fit_prior=True
    ))
elif model_name == 'PAC':
    model = OneVsRestClassifier(PassiveAggressiveClassifier(
        C=0.1,
        loss='hinge'
    ))


model.fit(train_data, train_types)

joblib.dump(model, f'models/{model_name}.pkl')

