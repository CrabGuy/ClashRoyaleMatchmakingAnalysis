import csv
import hashlib
import os
import random
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq
from pybloom_live import ScalableBloomFilter
import requests
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent

dotenv_path = PARENT_DIR / ".env"
load_dotenv(dotenv_path=dotenv_path)

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError(f"API_KEY not found in environment or file: {dotenv_path}")

DATA_DIR = PARENT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILENAME = DATA_DIR / "raw" / "trophy_battles.csv"
PARQUET_FILENAME = DATA_DIR / "processed" / "trophy_battles.parquet"

START_PLAYER_TAG = "%23PU2JQCUJQ"
AMOUNT_OF_BATTLES = 1_000
WORKERS = 5

HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def fetch_battlelog(player_tag):
    tag = player_tag if player_tag.startswith("%23") else f"%23{player_tag.lstrip('#')}"
    try:
        res = requests.get(
            f"https://proxy.royaleapi.dev/v1/players/{tag}/battlelog",
            headers=HEADERS,
            timeout=10,
        )
        if res.status_code == 200:
            return res.json()
        else:
            print(f"API Error [{res.status_code}] for tag {tag}: {res.text}", flush=True)
            return []
    except requests.exceptions.RequestException as e:
        print(f"Request exception for {tag}: {e}", flush=True)
        return []


def extract_battle_data(battle):
    if battle.get("type") != "PvP":
        return None

    team = battle.get("team", [{}])[0]
    opponent = battle.get("opponent", [{}])[0]
    p1_tag, p2_tag = team.get("tag", ""), opponent.get("tag", "")
    if not p1_tag or not p2_tag:
        return None

    timestamp = battle.get("battleTime", "")
    raw_id = f"{timestamp}_{'_'.join(sorted([p1_tag, p2_tag]))}"
    battle_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()

    p1_deck = "|".join(c["name"] for c in team.get("cards", []))
    p2_deck = "|".join(c["name"] for c in opponent.get("cards", []))

    p1_trophies = team.get("startingTrophies", 0) or 0
    p2_trophies = opponent.get("startingTrophies", 0) or 0
    avg_trophies = (p1_trophies + p2_trophies) / 2

    return {
        "battle_id": battle_id,
        "record": [battle_id, timestamp, avg_trophies, p1_deck, p2_deck],
        "opponent_tag": p2_tag,
    }


def filter_unseen(battles, bloom):
    def is_unseen(b):
        if b["battle_id"] in bloom:
            return False
        bloom.add(b["battle_id"])
        return True

    return list(filter(is_unseen, battles))


def next_tags(opponents, pool, count=WORKERS):
    def pick():
        if opponents and random.random() > 0.10:
            return random.choice(opponents)
        return random.choice(pool) if pool else START_PLAYER_TAG

    return [pick() for _ in range(count)]


def battle_stream(start_tag):
    bloom = ScalableBloomFilter(initial_capacity=1000, error_rate=0.001)
    pool = [start_tag]
    current_tags = [start_tag]

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        while True:
            results = executor.map(fetch_battlelog, current_tags)
            all_battles = [b for log in results for b in log]
            extracted = list(filter(None, map(extract_battle_data, all_battles)))
            new_battles = filter_unseen(extracted, bloom)

            yield from new_battles

            opponents = [b["opponent_tag"] for b in new_battles]
            if opponents:
                pool[:] = pool[-100:] + opponents

            current_tags = next_tags(opponents, pool)


def convert_csv_to_parquet(csv_file, parquet_file):
    reader = pv.open_csv(str(csv_file))
    writer = None
    for batch in reader:
        table = pa.Table.from_batches([batch])
        if writer is None:
            writer = pq.ParquetWriter(str(parquet_file), table.schema)
        writer.write_table(table)
    if writer:
        writer.close()


def save_battles(start_tag, target_count):
    with open(CSV_FILENAME, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["battle_id", "timestamp", "match_average_trophies", "player1_deck", "player2_deck"]
        )

        def write_and_log(item):
            count, battle = item
            writer.writerow(battle["record"])
            if count == 1 or count % 100 == 0:
                print(f"[{count}/{target_count}] Saved battle ID: {battle['record'][0]}", flush=True)

        list(map(write_and_log, enumerate(islice(battle_stream(start_tag), target_count), 1)))

    convert_csv_to_parquet(CSV_FILENAME, PARQUET_FILENAME)

    csv_mb = os.path.getsize(CSV_FILENAME) / (1024 * 1024)
    parquet_mb = os.path.getsize(PARQUET_FILENAME) / (1024 * 1024)
    print(f"CSV Size: {csv_mb:.2f} MB | Parquet Size: {parquet_mb:.2f} MB", flush=True)


if __name__ == "__main__":
    save_battles(START_PLAYER_TAG, AMOUNT_OF_BATTLES)