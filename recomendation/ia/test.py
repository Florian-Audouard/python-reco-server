import os
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

import pandas as pd

os.chdir(os.path.dirname(__file__))
# Load ratings
# df = pd.read_csv("data/ml-0.1m/ratings.csv")

# # Save only the required columns, tab-separated, no header
# df.to_csv("data/ml-0.1m/u.data", sep="\t", index=False, header=False)

reader = Reader(line_format="user item rating timestamp", sep="\t")
data = Dataset.load_from_file("data/ml-0.1m/u.data", reader=reader)


# Initialize the algorithm (SVD)
algo = SVD()

# Train the model on the full dataset
trainset = data.build_full_trainset()  # Train on the entire dataset
algo.fit(trainset)

for i in range(1, 50):
    prediction = algo.predict(str(1), str(i))
    print(f"Prediction for user {5} on movie {i}: {prediction.est:.2f}")
