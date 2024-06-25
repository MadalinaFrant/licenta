from .preprocess import preprocess_text


type_to_label = {
    'reliable': 0,
    'unreliable': 1,
    'political': 2,
    'clickbait': 3,
    'satire': 4,
    'bias': 5,
    'conspiracy': 6,
    'junksci': 7,
    'hate': 8,
    'rumor': 9,
    'fake': 10,
    'unknown': 11
}

label_to_type = {label: type for type, label in type_to_label.items()}


types = list(type_to_label.keys())


NUM_TYPES = len(types)

TRAIN_SIZE = 10000
TEST_SIZE = 2500

BATCH_SIZE = 32


def get_all_data(collection):
    content = []
    types = []
    for doc in collection.find():
        content.append(preprocess_text(doc['content']))
        types.append(type_to_label[doc['type']])
    return content, types


def get_data(collection, size):
    types_count = {}
    content = []
    types = []
    for doc in collection.find():
        if doc['type'] in types_count:
            if types_count[doc['type']] == size:
                continue
            types_count[doc['type']] += 1
        else:
            types_count[doc['type']] = 1
        content.append(preprocess_text(doc['content']))
        types.append(type_to_label[doc['type']])
    return content, types


def get_all_content_iterable(collection):
    for doc in collection.find():
        yield preprocess_text(doc['content'])


def get_content_iterable(collection, size):
    types_count = {}
    for doc in collection.find():
        if doc['type'] in types_count:
            if types_count[doc['type']] == size:
                continue
            types_count[doc['type']] += 1
        else:
            types_count[doc['type']] = 1
        yield preprocess_text(doc['content'])


def get_all_types(collection):
    types = []
    for doc in collection.find():
        types.append(type_to_label[doc['type']])
    return types


def get_types(collection, size):
    types_count = {}
    types = []
    for doc in collection.find():
        if doc['type'] in types_count:
            if types_count[doc['type']] == size:
                continue
            types_count[doc['type']] += 1
        else:
            types_count[doc['type']] = 1
        types.append(type_to_label[doc['type']])
    return types

