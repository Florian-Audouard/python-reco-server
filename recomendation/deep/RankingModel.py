import tensorflow as tf


class RankingModel(tf.keras.Model):
    def __init__(self, unique_userIds, unique_movieIds):
        super().__init__()

        embedding_dimension = 32
        self.user_embeddings = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(
                    vocabulary=unique_userIds, mask_token=None
                ),
                tf.keras.layers.Embedding(
                    len(unique_userIds) + 1,
                    embedding_dimension,
                    embeddings_initializer="random_normal",
                ),
            ]
        )

        self.product_embeddings = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(
                    vocabulary=unique_movieIds, mask_token=None
                ),
                tf.keras.layers.Embedding(
                    len(unique_movieIds) + 1,
                    embedding_dimension,
                    embeddings_initializer="random_normal",
                ),
            ]
        )

        self.ratings = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

    def call(self, userId, movieId):
        user_embeddings = self.user_embeddings(userId)
        product_embeddings = self.product_embeddings(movieId)
        return (
            self.ratings(tf.concat([user_embeddings, product_embeddings], axis=1)) * 5.0
        )
