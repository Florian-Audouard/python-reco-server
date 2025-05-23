z python-reco-server

(put it in preprocessing/data)

[ml-latest-small](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip) (rename it ml-0.1m)

[ml-1m](https://files.grouplens.org/datasets/movielens/ml-1m.zip)


### For run python server on windows
```bash
py -m uvicorn server:app --reload
```
### Get recommendations on windows (Account_id 2 is used in the example)
```bash
curl.exe http://localhost:8000/recommendations/2?top_n=10
```