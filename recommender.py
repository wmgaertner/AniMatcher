from datetime import datetime
import os, tempfile

from typing import Dict, Text
from pprint import pprint

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd

import tensorflow_recommenders as tfrs
from tensorflow_recommenders import metrics



# user_id int32
# anime_id int32
# rating int32
# watching_status int32
animelist_dataset = tf.data.experimental.make_csv_dataset(
    "data/filtered_list.csv",
    batch_size=1,
    num_epochs=1,
    header=True,
    select_columns=[0, 1, 2],
    column_defaults=[tf.string, tf.string, tf.int32]
)

# MAL_ID int32
# Name string
# Score string(possibly unrated)
# Genres string
anime_dataset =  tf.data.experimental.make_csv_dataset(
    "data/anime.csv",
    batch_size=1,
    num_epochs=1,
    header=True,
    select_columns=[0],
    column_defaults=[tf.string]
)

ratings = animelist_dataset.map(lambda x: {
    "user_id": x["user_id"],
    "MAL_ID": x["anime_id"],
    "rating": x["rating"]
}, num_parallel_calls=tf.data.experimental.AUTOTUNE)
anime = anime_dataset.map(lambda x: x["MAL_ID"], num_parallel_calls=tf.data.experimental.AUTOTUNE)

ratings = ratings.unbatch()
anime = anime.unbatch()


print(f"{datetime.now().strftime('%H:%M:%S:%f')} Shuffling data...")
tf.random.set_seed(53)
shuffled = ratings.shuffle(10_000_000, seed=53, reshuffle_each_iteration=True)

train = shuffled.take(47_664_000)
test = shuffled.skip(47_664_000).take(11916719)

# train = shuffled.take(150_000)
# test = shuffled.skip(150_000).take(20_000)



print(f"{datetime.now().strftime('%H:%M:%S:%f')} Batching data...")
anime_titles = anime.batch(1_000)
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"], num_parallel_calls=tf.data.experimental.AUTOTUNE)


print(f"{datetime.now().strftime('%H:%M:%S:%f')} Getting unique MAL_ID's...")
unique_anime_titles = np.unique(np.concatenate(list(anime_titles)))

print(f"{datetime.now().strftime('%H:%M:%S:%f')} Getting unique user_id's...")
unique_user_ids = pd.unique(np.concatenate(list(user_ids)))



class AniRecommenderModel(tfrs.models.Model):
    def __init__(self, rating_weight: float, retrieval_weight: float) -> None:
        super().__init__()

        embedding_dim = 32

        self.anime_model: tf.keras.layers.Layer = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_anime_titles, mask_token=None),
                tf.keras.layers.Embedding(len(unique_anime_titles)+1, embedding_dim)
        ])

        self.user_model: tf.keras.layers.Layer = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),
                tf.keras.layers.Embedding(len(unique_user_ids)+1, embedding_dim)
        ])

        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1),
        ])

        self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
        self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=anime.batch(1024).map(self.anime_model, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            )
        )

        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight

    def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
        user_embeddings = self.user_model(features["user_id"])
        anime_embeddings = self.anime_model(features["MAL_ID"])

        return ( 
            user_embeddings,
            anime_embeddings,

            self.rating_model(
                tf.concat([user_embeddings, anime_embeddings], axis=1)
            ),
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

        ratings = features.pop("rating")

        user_embeddings, anime_embeddings, rating_predictions = self(features)

        rating_loss = self.rating_task(
            labels=ratings,
            predictions=rating_predictions,
        )
        retrieval_loss = self.retrieval_task(user_embeddings, anime_embeddings)

        return (self.rating_weight * rating_loss + self.retrieval_weight * retrieval_loss)

print(f"{datetime.now().strftime('%H:%M:%S:%f')} Creating model...")
model = AniRecommenderModel(rating_weight=1.0, retrieval_weight=1.0)
print(f"{datetime.now().strftime('%H:%M:%S:%f')} Compiling model...")
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

print(f"{datetime.now().strftime('%H:%M:%S:%f')} Training model...")
cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

# save checkpoints
checkpoint_path = "training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    save_weights_only=True,
    verbose=0)

model.save_weights(checkpoint_path.format(epoch=0))
# print("Try to load checkpoint")
#load latest cp
# try:
#     latest = tf.train.latest_checkpoint(checkpoint_dir)
#     model.load_weights(latest)
#     print("Succesfully loaded checkpoint")
# except:
#     print("No checkpoint found")
#     pass

model.fit(cached_train, epochs=4, callbacks=[cp_callback], verbose=1)
print(f"{datetime.now().strftime('%H:%M:%S:%f')} Evaluating model...")
metrics = model.evaluate(cached_test, return_dict=True)
print(f"Retrieval top-100 accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}.")
print(f"Ranking RMSE: {metrics['root_mean_squared_error']:.3f}.")

index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
index.index_from_dataset(
    anime.batch(128).map(lambda mal_id: (mal_id, model.anime_model(mal_id)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
)

_, titles = index(tf.constant(["53"]))
print(f"Top 3 recommendations for '53': {titles[0, :3]}")

print(f"{datetime.now().strftime('%H:%M:%S:%f')} Saving model...")
# with tempfile.TemporaryDirectory() as tmp:
#     path = os.path.join(tmp, "model")
#     tf.saved_model.save(index, path)
#     loaded = tf.saved_model.load(path)

#     scores, titles = loaded(["53"])

#     print(f"Recommendations: {titles[0, :3]}")
# index.save('saved_model/my_model')
tf.saved_model.save(index, 'saved_model/my_model')