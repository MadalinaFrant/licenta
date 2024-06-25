import sys

sys.path.append('..')

import tensorflow as tf
import keras_tuner as kt

from sklearn.model_selection import train_test_split

import pymongo

from utils.utils import NUM_TYPES
from utils.utils import BATCH_SIZE
from utils.utils import get_data


client = pymongo.MongoClient('mongodb://localhost:27017/')

db = client['news_db']

train_collection = db['training_news']
test_collection = db['testing_news']


model_name = sys.argv[1]


size = 125
train_size = int(0.8 * size)
test_size = int(0.2 * size)

train_content, train_types = get_data(train_collection, train_size)
test_content, test_types = get_data(test_collection, test_size)


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


def CNN_model_builder(hp):
    model = tf.keras.Sequential([
        vectorize_layer,
        tf.keras.layers.Embedding(
            len(vectorize_layer.get_vocabulary()), 
            hp.Int('embedding_dim', 50, 100, step=50)
        ),
        tf.keras.layers.Conv1D(
            hp.Int('filters', 32, 128, step=32),
            hp.Int('kernel_size', 2, 6, step=2),
            activation='relu'
        ),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(
            hp.Int('units', 32, 64, step=32),
            activation='relu'
        ),
        tf.keras.layers.Dropout(
            hp.Float('dropout', 0.1, 0.5, step=0.1)
        ),
        tf.keras.layers.Dense(NUM_TYPES, activation='softmax')
    ])

    return model


def RNN_model_builder(hp):
    model = tf.keras.Sequential([
        vectorize_layer,
        tf.keras.layers.Embedding(
            len(vectorize_layer.get_vocabulary()), 
            hp.Int('embedding_dim', 50, 100, step=50)
        ),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                hp.Int('LSTM_units', 32, 64, step=32)
            )
        ),
        tf.keras.layers.Dense(
            hp.Int('units', 32, 64, step=32),
            activation='relu'
        ),
        tf.keras.layers.Dropout(
            hp.Float('dropout', 0.1, 0.5, step=0.1)
        ),
        tf.keras.layers.Dense(NUM_TYPES, activation='softmax')
    ])

    return model


if model_name == 'CNN':
    model_builder = CNN_model_builder
elif model_name == 'RNN':
    model_builder = RNN_model_builder

model_builder.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['categorical_accuracy']
)

tuner = kt.Hyperband(
    model_builder,
    objective='val_categorical_accuracy',
    max_epochs=10,
    directory='kt_dir',
    project_name='news_classification'
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

tuner.search(
    train_ds,
    validation_data=val_ds,
    callbacks=[early_stopping, model_checkpoint]
)


best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

if model_name == 'CNN':

    print(f'embedding_dim: {best_hps.get("embedding_dim")}')
    print(f'filters: {best_hps.get("filters")}')
    print(f'kernel_size: {best_hps.get("kernel_size")}')
    print(f'units: {best_hps.get("units")}')
    print(f'dropout: {best_hps.get("dropout")}')

elif model_name == 'RNN':

    print(f'embedding_dim: {best_hps.get("embedding_dim")}')
    print(f'LSTM_units: {best_hps.get("LSTM_units")}')
    print(f'units: {best_hps.get("units")}')
    print(f'dropout: {best_hps.get("dropout")}')

model = tuner.hypermodel.build(best_hps)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stopping, model_checkpoint]
)

loss, accuracy = model.evaluate(test_ds)

print('Loss: ', loss)
print('Accuracy: ', accuracy)


model.save(f'models/best_{model_name}.keras')

