import sys

sys.path.append('..')

from sklearn.feature_extraction.text import TfidfVectorizer

import joblib
import pymongo

from utils.utils import TRAIN_SIZE, TEST_SIZE
from utils.utils import get_content_iterable


client = pymongo.MongoClient('mongodb://localhost:27017/')

db = client['news_db']

train_collection = db['training_news']
test_collection = db['testing_news']


tfidf = TfidfVectorizer(
    min_df=2,
    max_df=0.5,
    ngram_range=(1, 2),
    sublinear_tf=True,
    norm='l2'
)


train_content = get_content_iterable(train_collection, TRAIN_SIZE)
train_data = tfidf.fit_transform(train_content)

test_content = get_content_iterable(test_collection, TEST_SIZE)
test_data = tfidf.transform(test_content)


joblib.dump(tfidf, 'models/TFIDF.pkl')

joblib.dump(train_data, 'models/train_data.pkl')
joblib.dump(test_data, 'models/test_data.pkl')
