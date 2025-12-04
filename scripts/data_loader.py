import tensorflow as tf
from tensorflow import keras
import polars as pl
import numpy as np
import pyarrow.parquet as pq
import random

DATA_PATH = "data/features/engineered_features.parquet"

BATCH_SIZE = 4096*8

NUMERICAL_FEATURES = [
    'trip_distance',
    'trip_duration',
    'route_mean_distance',
    'distance_ratio',
    'avg_speed'
]

BINARY_FEATURES = [
    'is_weekend',
    'is_rush_hour'
]

TARGET_COLUMN = 'fare_amount'

EMBEDDING_FEATURES = {
    'travel_year': {'vocab_size': 10, 'embed_dim': 3},      # Adjust vocab_size based on your data
    'travel_month': {'vocab_size': 12, 'embed_dim': 4},
    'travel_day': {'vocab_size': 31, 'embed_dim': 6},
    'travel_hour': {'vocab_size': 24, 'embed_dim': 6},
    'travel_weekday': {'vocab_size': 7, 'embed_dim': 3},
    'passenger_count': {'vocab_size': 10, 'embed_dim': 3},
    'PULocationID': {'vocab_size': 265, 'embed_dim': 20},   # NYC has ~265 taxi zones
    'DOLocationID': {'vocab_size': 265, 'embed_dim': 20}
}



def train_test_split(dataset_path, val_size=0.15, test_size=0.15):
    df = pl.read_parquet(dataset_path)
    num_samples = len(df)
    val_samples = int(num_samples * val_size)
    test_samples = int(num_samples * test_size)
    train_samples = num_samples - val_samples - test_samples

    df_test = df.sample(fraction=test_size, shuffle=True, with_replacement=False)
    df_val = df.sample(fraction=val_size, shuffle=True, with_replacement=False)
    df_train = df

    df_test.write_parquet('data/processed/test.parquet')
    df_val.write_parquet('data/processed/val.parquet')
    df_train.write_parquet('data/processed/train.parquet')




def preprocess_categorical(batch):
    if 'travel_month' in batch.columns and batch['travel_month'].min() >= 1:
        batch['travel_month'] -= 1
    
    if 'travel_day' in batch.columns and batch['travel_day'].min() >= 1:
        batch['travel_day'] -= 1
    
    if 'passenger_count' in batch.columns and batch['passenger_count'].min() >= 1:
        batch['passenger_count'] -= 1

    if 'PULocationID' in batch.columns:
        batch['PULocationID'] = batch['PULocationID'].astype(int)
    
    if 'DOLocationID' in batch.columns:
        batch['DOLocationID'] = batch['DOLocationID'].astype(int)
    
    return batch
    


def normalize_data(file_path, numerical_features, sample_size=1_000_000):
    df = pl.read_parquet(file_path).sample(sample_size, shuffle=True)
    normalization_layer = keras.layers.Normalization(axis=-1)
    normalization_layer.adapt(df[numerical_features].to_numpy())
    return normalization_layer

def create_dataset(file_path, batch_size=BATCH_SIZE, shuffle=True, buffer_size=BATCH_SIZE  *5):
    def data_generator():
        df = pq.ParquetFile(file_path)
        num_rows = df.metadata.num_rows

        processed_rows = 0
        batch_idx = 0

        for batch in df.iter_batches(batch_size=1000000):
            
            batch = batch.to_pandas()
            batch = preprocess_categorical(batch)
            
            batch = batch.dropna(subset=[TARGET_COLUMN])

            embedding_data = {}

            for feature in EMBEDDING_FEATURES.keys():
                embedding_data[feature] = batch[feature].values
        
            numerical_data = batch[NUMERICAL_FEATURES].astype(np.float32).values
            binary_data = batch[BINARY_FEATURES].astype(np.float32).values

            y = batch[TARGET_COLUMN].astype(np.float32).values

            processed_rows += len(batch)
            batch_idx += 1

            for i in range(0, len(y), batch_size):
                end_idx = min(i + batch_size, len(y))
                features = {
                    **{col: embedding_data[col][i:end_idx] for col in EMBEDDING_FEATURES.keys()},
                    'numerical_input': numerical_data[i:end_idx],
                    'binary_input': binary_data[i:end_idx]
                }

                yield features, y[i:end_idx]

    output_signature = (
        {
            **{col: tf.TensorSpec(shape=(None,), dtype=tf.int32) for col in EMBEDDING_FEATURES.keys()},
            'numerical_input': tf.TensorSpec(shape=(None, len(NUMERICAL_FEATURES)), dtype=tf.float32),
            'binary_input': tf.TensorSpec(shape=(None, len(BINARY_FEATURES)), dtype=tf.int32)
        },
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=output_signature
    )

    if shuffle:
        dataset = dataset.unbatch()
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


            
if __name__ == '__main__':
    train_test_split(DATA_PATH)
    
