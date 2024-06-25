import csv
import random
import pymongo


client = pymongo.MongoClient("mongodb://localhost:27017/")

db = client["news_db"]

collection = db["news"]


filename = "../dataset/corpus_news.csv"

csv.field_size_limit(10 ** 6)


def insert_batch(batch):
    random.shuffle(batch)
    collection.insert_many(batch)


batch_size = 1000

with open(filename, 'r', newline='') as csv_file:

    csv_reader = csv.DictReader(csv_file)

    batch = []

    for row in csv_reader:
        if not row.get('type') or not row.get('content'):
            continue
        else:
            doc = {
                'content': row.get('content'),
                'type': row.get('type')
            }

            batch.append(doc)

            if len(batch) == batch_size:
                insert_batch(batch)
                batch = []
    
    if batch:
        insert_batch(batch)
