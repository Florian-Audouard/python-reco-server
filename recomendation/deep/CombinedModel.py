from typing import Dict, Text
import tensorflow as tf
import tensorflow_recommenders as tfrs


class CombinedModel(tfrs.models.Model):
    def __init__(
        self, user_model, movie_model, rating_model, retrieval_task, ranking_task
    ):
        super().__init__()
        self.user_model = user_model
        self.movie_model = movie_model
        self.rating_model = rating_model
        self.retrieval_task = retrieval_task
        self.rating_task = ranking_task

    def compute_loss(
        self, features: Dict[Text, tf.Tensor], training=False
    ) -> tf.Tensor:
        user_embeddings = self.user_model(features["userId"])
        movie_embeddings = self.movie_model(features["movieId"])
        rating_predictions = self.rating_model(
            tf.concat([user_embeddings, movie_embeddings], axis=1)
        )

        # Combine both retrieval and ranking tasks
        return self.retrieval_task(
            user_embeddings, movie_embeddings
        ) + self.rating_task(labels=features["rating"], predictions=rating_predictions)
