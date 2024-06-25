import sys

sys.path.append('..')

import tensorflow as tf

from sklearn.model_selection import train_test_split

from utils.utils import NUM_TYPES
from utils.utils import TRAIN_SIZE, TEST_SIZE
from utils.utils import BATCH_SIZE
from utils.utils import get_data

import pymongo

client = pymongo.MongoClient('mongodb://localhost:27017/')

db = client['news_db']

train_collection = db['training_news']
test_collection = db['testing_news']


model_name = sys.argv[1]


train_content, train_types = get_data(train_collection, TRAIN_SIZE)
test_content, test_types = get_data(test_collection, TEST_SIZE)

train_content, val_content, train_types, val_types = train_test_split(
    train_content, train_types, test_size=0.2, stratify=train_types
)

train_ds = tf.data.Dataset.from_tensor_slices((train_content, train_types))
val_ds = tf.data.Dataset.from_tensor_slices((val_content, val_types))
test_ds = tf.data.Dataset.from_tensor_slices((test_content, test_types))


vectorize_layer = tf.keras.layers.TextVectorization(
    standardize=None,
    max_tokens=1000000,
    output_mode='int',
    ngrams=(1, 2)
)

train_text = train_ds.map(lambda x, y: x)

vectorize_layer.adapt(train_text)


train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, NUM_TYPES)))
val_ds = val_ds.map(lambda x, y: (x, tf.one_hot(y, NUM_TYPES)))
test_ds = test_ds.map(lambda x, y: (x, tf.one_hot(y, NUM_TYPES)))


train_ds = train_ds.cache('./train_cache')
val_ds = val_ds.cache('./val_cache')
test_ds = test_ds.cache('./test_cache')

train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=True)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=True)

train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)


if model_name == 'CNN':

    model = tf.keras.Sequential([
        vectorize_layer,
        tf.keras.layers.Embedding(len(vectorize_layer.get_vocabulary()), 100),
        tf.keras.layers.Conv1D(128, 4, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(NUM_TYPES, activation='softmax')
    ])

elif model_name == 'RNN':

    model = tf.keras.Sequential([
        vectorize_layer,
        tf.keras.layers.Embedding(len(vectorize_layer.get_vocabulary()), 100),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(NUM_TYPES, activation='softmax')
    ])


model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['categorical_accuracy']
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    restore_best_weights=True
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.keras', 
    monitor='val_loss', 
    save_best_only=True
)

epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs, 
    callbacks=[early_stopping, model_checkpoint]
)


loss, accuracy = model.evaluate(test_ds)

print('Loss: ', loss)
print('Accuracy: ', accuracy)


model.save(f'models/{model_name}.keras')

