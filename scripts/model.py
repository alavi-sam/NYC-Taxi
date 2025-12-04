import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

from data_loader import create_dataset, normalize_data, preprocess_categorical
from data_loader import EMBEDDING_FEATURES, NUMERICAL_FEATURES, BINARY_FEATURES, DATA_PATH

import pickle


VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15
EPOCHS = 10
SAVE_PATH = '../models/taxi_fare_model.keras'
NORMALIZER_PATH = '../models/normalizer.pkl'




def build_model(normalizer):
    embedding_layers = []
    embedding_inputs = []

    for col_name, config in EMBEDDING_FEATURES.items():
        inp = keras.layers.Input(shape=(1,), name=col_name, dtype=tf.int32)

        emb = keras.layers.Embedding(
            input_dim=config['vocab_size'],
            output_dim=config['embed_dim'],
            name=f'{col_name}_embedding'
        )(inp)

        emb = keras.layers.Flatten()(emb)

        embedding_inputs.append(inp)
        embedding_layers.append(emb)

    numerical_inp = keras.layers.Input(
        shape=(len(NUMERICAL_FEATURES),),
        name='numerical_input',
        dtype=tf.float32
    )

    normalized_numerical = normalizer(numerical_inp)

    binary_inp = keras.layers.Input(
        shape=(len(BINARY_FEATURES),),
        name='binary_input',
        dtype=tf.float32
    )

    all_features = keras.layers.concatenate(
        embedding_layers + [normalized_numerical, binary_inp]
    )

    x = keras.layers.Dense(256, activation='relu')(all_features)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(1, activation='linear')(x)

    model = keras.Model(
        inputs=embedding_inputs + [numerical_inp, binary_inp],
        outputs=output
    )
    
    return model


def train_model(save_path=SAVE_PATH, epochs=EPOCHS):

    normalizer = normalize_data(DATA_PATH, NUMERICAL_FEATURES)

    model = build_model(normalizer)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mean_absolute_error', 'mean_absolute_percentage_error'],
    )

    model.summary()

    import pyarrow.parquet as pq
    parquet_file = pq.ParquetFile(DATA_PATH)
    num_samples = parquet_file.metadata.num_rows
    print(f'Total samples in dataset: {num_samples:,}')

    dataset = create_dataset(
        file_path=DATA_PATH,
    )

    val_samples = int(num_samples * VALIDATION_SIZE)
    test_samples = int(num_samples * TEST_SIZE)
    train_samples = num_samples - val_samples - test_samples

    train_dataset = dataset.take(train_samples).repeat()
    val_dataset = dataset.skip(train_samples).take(val_samples).repeat()
    test_dataset = dataset.skip(train_samples + val_samples)

    callbacks = [
    keras.callbacks.ModelCheckpoint(
        save_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        verbose=1
    ),
    keras.callbacks.ProgbarLogger()
    ]

    print('start training!')

    steps_per_epoch = train_samples // 2048
    validation_steps = val_samples // 2048

    print('steps_per_epoch:', steps_per_epoch)
    print('validation_steps:', validation_steps)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=1
    )

    test_steps = test_samples // 2048
    mse, mae, mape = model.evaluate(test_dataset, steps=test_steps)
    print('Mean Squared Error:', mse)
    print('Mean Absolute Error:', mae)
    print('Mean Absolute Percentage Error:', mape)

    print('saving final model!')
    model.save(save_path)

    with open(NORMALIZER_PATH, 'wb') as f:
        pickle.dump(normalizer, f)
    print(f'Normalizer saved to {NORMALIZER_PATH}!')

    return model, history



def predict(model, data):
    data = preprocess_categorical(data)

    with open(NORMALIZER_PATH, 'rb') as f:
        normalizer = pickle.load(f)

    data[NUMERICAL_FEATURES] = normalizer(data[NUMERICAL_FEATURES])

    input = {
        **{col: data[col].values for col in EMBEDDING_FEATURES.keys()},
        'numerical_input': data[NUMERICAL_FEATURES].values,
        'binary_input': data[BINARY_FEATURES].values
    }

    predictions = model.predict(input)
    return predictions.flatten()


if __name__ == '__main__':
    model, history = train_model()

    sample_data = pd.read_parquet(DATA_PATH, nrows=10)
    predictions = predict(model, sample_data)

    print('sample predictions:')
    print('actual fares:', sample_data['fare_amount'].values)
    print('predicted fares:', predictions)




    


    