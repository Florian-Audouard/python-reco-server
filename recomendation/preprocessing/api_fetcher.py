import os
from dotenv import load_dotenv
import requests
import json
import threading
from tqdm import tqdm
import logging
import pycountry
import re
import csv

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
    res["Runtime"] = "0"
    res["Genre"] = genres
    res["Plot"] = "Unknown"
    res["Poster"] = "https://upload.wikimedia.org/wikipedia/en/6/60/No_Picture.jpg"
    res["Country"] = "Unknown"
    res["Director"] = "Unknown"
    res["Writer"] = "Unknown"
    res["Actors"] = "Unknown"
    return res


def invalid_data(data):
    return data is None or data == "N/A" or data == ""


def check_dict(checking_dict: dict):
    if invalid_data(checking_dict.get("Released")):
        checking_dict["Released"] = "Unknown"
    if invalid_data(checking_dict.get("Runtime")):
        checking_dict["Runtime"] = "0"
    if invalid_data(checking_dict.get("Genre")):
        checking_dict["Genre"] = "Unknown"
    if invalid_data(checking_dict.get("Plot")):
        checking_dict["Plot"] = "Unknown"
    if invalid_data(checking_dict.get("Poster")):
        checking_dict["Poster"] = (
            "https://upload.wikimedia.org/wikipedia/en/6/60/No_Picture.jpg"
        )
    if invalid_data(checking_dict.get("Country")):
        checking_dict["Country"] = "Unknown"
    if invalid_data(checking_dict.get("Director")):
        checking_dict["Director"] = "Unknown"
    if invalid_data(checking_dict.get("Writer")):
        checking_dict["Writer"] = "Unknown"
    if invalid_data(checking_dict.get("Actors")):
        checking_dict["Actors"] = "Unknown"

    if "Runtime" in checking_dict and "S" in checking_dict["Runtime"]:
        checking_dict["Runtime"] = checking_dict["Runtime"].replace("S", "")

    for key in checking_dict:
        if isinstance(checking_dict[key], str):
            checking_dict[key] = checking_dict[key].replace('"', "")
            checking_dict[key] = checking_dict[key].replace("\n", "")
            checking_dict[key] = checking_dict[key].replace("  ", " ")
            checking_dict[key] = checking_dict[key].strip()


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
    res["Title"] = res_json.get("title", "")
    res["Released"] = res_json.get("release_date", "")
    res["Runtime"] = str(res_json.get("runtime", "0"))
    list_genre = [g["name"] for g in res_json.get("genres", [])]
    res["Genre"] = ",".join(list_genre)
    res["Plot"] = res_json.get("overview", "")
    if res_json.get("poster_path") is None:
        res["Poster"] = "https://upload.wikimedia.org/wikipedia/en/6/60/No_Picture.jpg"
    else:
        res["Poster"] = "https://image.tmdb.org/t/p/w500" + res_json["poster_path"]
    country = res_json["origin_country"]
    if len(country) == 0:
        res["Country"] = "Unknown"
    else:
        try:
            code = country[0]
            res["Country"] = pycountry.countries.get(alpha_2=code).name
        except Exception as e:
            if code == "SU":
                res["Country"] = "Soviet Union"
            elif code == "YU":
                res["Country"] = "Yugoslavia"
            else:
                res["Country"] = "Unknown"

    res_json = my_request(URL_MOVIE + f"/{id_movie}/credits", params=params, timeout=10)
    res["Director"] = next(
        (m["name"] for m in res_json.get("crew", []) if m["job"] == "Director"),
        "Unknown",
    )
    list_writer = [
        m["name"] for m in res_json.get("crew", []) if m["job"] == "Screenplay"
    ]
    res["Writer"] = ",".join(list_writer)
    list_actors = [m["name"] for m in res_json.get("cast", [])]
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


def extract_title_and_year(raw_title):
    raw_title = raw_title.strip()

    # Extract year from the end
    year_match = re.search(r"\((\d{4})\)\s*$", raw_title)
    year = year_match.group(1) if year_match else None

    # Remove the year from the end
    if year:
        raw_title = re.sub(r"\s*\(\d{4}\)\s*$", "", raw_title)

    # Remove leading (number) like (500)
    raw_title = re.sub(r"^\([^)]*\)\s*", "", raw_title)

    # Remove all remaining parenthetical content
    raw_title = re.sub(r"\([^)]*\)", "", raw_title)

    # Final cleanup
    clean_title = raw_title.strip().strip('"').strip()

    return clean_title, year


results_lock = threading.Lock()
csv_results = []
json_results = []

# Read CSV with DictReader and chunk rows
with open(FULL_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

chunks = [[] for _ in range(num_api_keys)]
for i, row in enumerate(rows):
    chunks[i % num_api_keys].append(row)


def process_chunk(chunk, api_key):
    local_csv = []
    local_json = []

    for row in tqdm(chunk, desc=f"Thread {api_key[:4]}..."):
        movie_id = row["movieId"]
        raw_title = row["title"]
        genres = row.get("genres", "")

        title, years = extract_title_and_year(raw_title)

        movie_data = fetch_movie_data(title, years, api_key)
        if movie_data.get("Response") == "False":
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


# Start threads
threads = []
for i in range(num_api_keys):
    t = threading.Thread(target=process_chunk, args=(chunks[i], api_key_list[i]))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

csv_results = sorted(csv_results, key=lambda x: int(x.split(",")[0]))
json_results = sorted(json_results, key=lambda x: int(x["MovieId"]))

header = (
    "movieId,title,released,runtime,genre,director,writer,actors,plot,country,poster\n"
)

# Écriture des fichiers
with open(CREATED_PATH_CSV, "w", encoding="utf-8") as f:
    f.write(header)
    f.writelines(line + "\n" for line in csv_results)

with open(CREATED_PATH_JSON, "w", encoding="utf-8") as f:
    json.dump(json_results, f, indent=4, ensure_ascii=False)
