import os
from dotenv import load_dotenv
import requests
import json
import threading
from tqdm import tqdm
import logging
import pycountry

URL_SEARCH = "https://api.themoviedb.org/3/search/movie"
URL_MOVIE = "https://api.themoviedb.org/3/movie"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("report.log", mode="w")],
)
log = logging.getLogger(__name__)
URL = "http://www.omdbapi.com/"

FOLDER = "ml-0.1m"
MOVIE_FILENAME = "movies.csv"
CREATED_FILENAME_CSV = "movies_created.csv"
CREATED_PATH_JSON = "movies_created.json"
os.chdir(os.path.dirname(__file__))
load_dotenv()
api_key_str = os.getenv("API_KEY")
api_key_themoviedb = os.getenv("API_KEY_THEMOVIEDB")
api_key_list = api_key_str.split(",")
num_api_keys = len(api_key_list)
FULL_PATH = os.path.join("data", FOLDER, MOVIE_FILENAME)
CREATED_PATH_CSV = os.path.join("data", FOLDER, CREATED_FILENAME_CSV)
CREATED_PATH_JSON = os.path.join("data", FOLDER, CREATED_PATH_JSON)


def my_request(url, params, timeout):

    while True:
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            continue
        except requests.exceptions.RequestException:
            continue


def test_api_keys(api_keys_list):
    test_title = "Inception"
    print("Testing API keys...")
    for api_key in api_keys_list:
        params = {"t": test_title, "apikey": api_key}
        try:
            my_request(URL, params, timeout=10)
            print(f"✅ API key {api_key} is valid.")
        except requests.RequestException as e:
            print(f"❌ API key {api_key} caused an exception: {e}")
            return False
    return True


if not test_api_keys(api_key_list):
    print("a key is invalid or exhausted, please check your API keys.")
    exit(1)


def fetch_movie_data(title, years, api_key):
    params = {"t": title, "y": years, "apikey": api_key}
    response = my_request(URL, params, timeout=10)
    return response


def empty_dict(movie_id, title, genres):
    log.error("No results found for movie : (%s) title: %s", movie_id, title)
    res = dict()
    res["Title"] = title
    res["Released"] = "Unknown"
    res["Runtime"] = "0 min"
    res["Genre"] = genres
    res["Plot"] = "Unknown"
    res["Poster"] = "https://upload.wikimedia.org/wikipedia/en/6/60/No_Picture.jpg"
    res["Country"] = "Unknown"
    res["Director"] = "Unknown"
    res["Writer"] = "Unknown"
    res["Actors"] = "Unknown"
    return res


def check_dict(checking_dict: dict):
    if checking_dict["Released"] is None:
        checking_dict["Released"] = "Unknown"
    if checking_dict["Runtime"] is None:
        checking_dict["Runtime"] = "0"
    if checking_dict["Genre"] is None:
        checking_dict["Genre"] = "Unknown"
    if checking_dict["Plot"] is None:
        checking_dict["Plot"] = "Unknown"
    if checking_dict["Poster"] is None:
        checking_dict["Poster"] = (
            "https://upload.wikimedia.org/wikipedia/en/6/60/No_Picture.jpg"
        )
    if checking_dict["Country"] is None:
        checking_dict["Country"] = "Unknown"
    if checking_dict["Director"] is None:
        checking_dict["Director"] = "Unknown"
    if checking_dict["Writer"] is None:
        checking_dict["Writer"] = "Unknown"
    if checking_dict["Actors"] is None:
        checking_dict["Actors"] = "Unknown"


def custom_themoviedb_search(movie_id, title, genres, api_key):
    params = {
        "api_key": api_key,
        "query": title,
    }
    res_json = my_request(URL_SEARCH, params=params, timeout=10)
    if len(res_json["results"]) == 0:
        return empty_dict(movie_id, title, genres)
    id_movie = res_json["results"][0]["id"]
    params = {
        "api_key": api_key,
    }
    res_json = my_request(URL_MOVIE + f"/{id_movie}", params=params, timeout=10)

    res = dict()
    res["Title"] = res_json["title"]
    res["Released"] = res_json["release_date"]
    res["Runtime"] = str(res_json["runtime"])
    list_genre = [g["name"] for g in res_json["genres"]]
    res["Genre"] = ",".join(list_genre)
    res["Plot"] = res_json["overview"]
    if res_json["poster_path"] is None:
        res["Poster"] = "https://upload.wikimedia.org/wikipedia/en/6/60/No_Picture.jpg"
    else:
        res["Poster"] = "https://image.tmdb.org/t/p/w500" + res_json["poster_path"]
    code = res_json["origin_country"][0]
    try:
        res["Country"] = pycountry.countries.get(alpha_2=code).name
    except Exception as e:
        if code == "SU":
            res["Country"] = "Soviet Union"
        elif code == "YU":
            res["Country"] = "Yugoslavia"
        else:
            log.error("Country not found for code %s: %s", code, str(e))
            res["Country"] = "Unknown"

    res_json = my_request(URL_MOVIE + f"/{id_movie}/credits", params=params, timeout=10)
    res["Director"] = next(
        (m["name"] for m in res_json["crew"] if m["job"] == "Director"), None
    )
    list_writer = [m["name"] for m in res_json["crew"] if m["job"] == "Screenplay"]
    res["Writer"] = ",".join(list_writer)
    list_actors = [m["name"] for m in res_json["cast"]]
    res["Actors"] = ",".join(list_actors)

    return res


def create_line(data):
    res = [
        data["MovieId"],
        f'"{data["Title"]}"',
        data["Released"],
        data["Runtime"].replace(" min", ""),
        data["Genre"].replace(",", "|"),
        data["Director"].replace(",", "|"),
        data["Writer"].replace(",", "|"),
        data["Actors"].replace(",", "|"),
        f'"{data["Plot"]}"',
        data["Country"].replace(",", "|"),
        data["Poster"],
    ]
    return ",".join(res)


def create_json(data):
    res = {
        "MovieId": data["MovieId"],
        "Title": data["Title"],
        "Released": data["Released"],
        "Runtime": data["Runtime"].replace(" min", ""),
        "Genre": data["Genre"].replace(",", "|"),
        "Director": data["Director"].replace(",", "|"),
        "Writer": data["Writer"].replace(",", "|"),
        "Actors": data["Actors"].replace(",", "|"),
        "Plot": data["Plot"],
        "Country": data["Country"].replace(",", "|"),
        "Poster": data["Poster"],
    }
    return res


# Lecture du fichier
with open(FULL_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

header = (
    "movieId,title,released,runtime,genre,director,writer,actors,plot,country,poster\n"
)
movie_lines = lines[1:]

# Découpage en lots
chunks = [[] for _ in range(num_api_keys)]
for i, line in enumerate(movie_lines):
    chunks[i % num_api_keys].append(line)  # +1 à cause de l'en-tête

results_lock = threading.Lock()
csv_results = []
json_results = []


def process_chunk(chunk, api_key):
    local_csv = []
    local_json = []

    for line in tqdm(chunk, desc=f"Thread {api_key[:4]}..."):
        line = line.rstrip()
        if not line:
            continue
        title = line.split(",")[1].split("(")[0].strip()
        years = line.split(",")[1].split("(")[-1].split(")")[0].strip()
        movie_data = fetch_movie_data(title, years, api_key)
        movie_id = line.split(",")[0]
        if movie_data.get("Response") == "False":
            genres = line.split(",")[-1]
            movie_data = custom_themoviedb_search(
                movie_id, title, genres, api_key_themoviedb
            )

        movie_data["MovieId"] = movie_id
        check_dict(movie_data)
        local_csv.append(create_line(movie_data))
        local_json.append(movie_data)

    with results_lock:
        csv_results.extend(local_csv)
        json_results.extend(local_json)


# Lancement des threads
threads = []
for i in range(num_api_keys):
    t = threading.Thread(target=process_chunk, args=(chunks[i], api_key_list[i]))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

csv_results = sorted(csv_results, key=lambda x: int(x.split(",")[0]))
json_results = sorted(json_results, key=lambda x: int(x["MovieId"]))

# Écriture des fichiers
with open(CREATED_PATH_CSV, "w", encoding="utf-8") as f:
    f.write(header)
    f.writelines(line + "\n" for line in csv_results)

with open(CREATED_PATH_JSON, "w", encoding="utf-8") as f:
    json.dump(json_results, f, indent=4, ensure_ascii=False)
