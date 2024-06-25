import pymongo


client = pymongo.MongoClient('mongodb://localhost:27017/')

db = client["news_db"]

collection = db["news"]

train_collection = db["training_news"]
test_collection = db["testing_news"]


def get_types_count(collection):
    types = {}

    for doc in collection.find():
        if doc['type'] in types:
            types[doc['type']] += 1
        else:
            types[doc['type']] = 1

    return types


types = get_types_count(collection)
sorted_types = sorted(types.items(), key=lambda x: x[1], reverse=True)
for type_key, count in sorted_types:
    print(f"{type_key}: {count}")
print()


types = get_types_count(train_collection)
for type_key in types:
    print(f"{type_key}: {types[type_key]}")
print()


types = get_types_count(test_collection)
for type_key in types:
    print(f"{type_key}: {types[type_key]}")
print()

