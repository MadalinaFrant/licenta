import sys

sys.path.append('..')

import random
import pymongo

from utils.utils import types


client = pymongo.MongoClient('mongodb://localhost:27017/')

db = client["news_db"]

collection = db["news"]

train_collection = db["training_news"]
test_collection = db["testing_news"]


BALANCED_SIZE = 75000

train_size = int(0.8 * BALANCED_SIZE)
test_size = int(0.2 * BALANCED_SIZE)

batch_size = 1000

def insert_in_batches(docs, collection, batch_size):
    batch = []
    for doc in docs:
        batch.append(doc)
        if len(batch) == batch_size:
            collection.insert_many(batch)
            batch = []

    if batch:
        collection.insert_many(batch)


train_docs = []
for type_name in types:
    docs = list(collection.find({"type": type_name}).limit(train_size))
    train_docs += docs

random.shuffle(train_docs)
insert_in_batches(train_docs, train_collection, batch_size)


test_docs = []
for type_name in types:
    docs = list(collection.find({"type": type_name}).skip(train_size).limit(test_size))
    test_docs += docs

random.shuffle(test_docs)
insert_in_batches(test_docs, test_collection, batch_size)

