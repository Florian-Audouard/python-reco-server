import tensorflow as tf
import tensorflow_recommenders as tfrs
from RankingModel import RankingModel


class MovieModel(tfrs.models.Model):
    def __init__(self, unique_userIds, unique_movieIds):
        super().__init__()
        self.ranking_model: tf.keras.Model = RankingModel(
            unique_userIds, unique_movieIds
        )
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )

    def compute_loss(self, features, training=False):
        rating_predictions = self.ranking_model(features["userId"], features["movieId"])
        return self.task(labels=features["rating"], predictions=rating_predictions)
