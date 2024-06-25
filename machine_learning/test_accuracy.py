import sys

sys.path.append('..')

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

import joblib
import pymongo

from utils.utils import TEST_SIZE
from utils.utils import types
from utils.utils import get_types


client = pymongo.MongoClient('mongodb://localhost:27017/')

db = client['news_db']

test_collection = db['testing_news']


model_name = sys.argv[1]

model = joblib.load(f'models/{model_name}.pkl')


test_data = joblib.load('models/test_data.pkl')
test_types = get_types(test_collection, TEST_SIZE)


pred = model.predict(test_data)


print(classification_report(test_types, pred, target_names=types))

acc = accuracy_score(test_types, pred)
print(f'Accuracy: {round(acc * 100, 2)}%')

cm = confusion_matrix(test_types, pred)

plt.figure(figsize=(16, 9), dpi=1000)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=types, yticklabels=types)
plt.ylabel('Prediction', fontsize=12)
plt.xlabel('Actual', fontsize=12)
plt.title('Confusion Matrix', fontsize=16)
plt.savefig(f'images/{model_name}_confusion_matrix.png')

