import os
from dotenv import load_dotenv
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
import requests
import time
import json

# DEPRECATED

load_dotenv()

RIOT_API_KEY = os.getenv("RIOT_API_KEY").strip()
if not RIOT_API_KEY:
    raise SystemExit("Missing RIOT_API_KEY, env not recognized.")

HEADERS = {"X-Riot-Token": RIOT_API_KEY}

REGION = "na1"
MATCH_REGION = "americas"
QUEUE_ID_RANKED_SOLO = 420

DB_PATH = "riot_lol.sqlite"

SEED_RIOT_IDS = [
    # ("BenGi", "9281"),
    # ("whkpty", "NA1"),
    # ("Psylore", "NA1"),
    # ("Findy", "NA1"),
    # ("Levi", "FEL"),
    
    # # ////////////////////////
    
    # ("barbagianni", "3209"), # diamond
    # ("Pata", "GOAT"),
    # ("Heretic", "LFN"),
    # ("peewee10763", "NA1"), # Gold
    # ("Zenyo", "NA1"),
    # ("Kenzo Ryosuke", "NA1"),
    # ("Forceripe", "NA1")
    
    # # ////////////////////////////
    
    ("Dpena", "DAD"),
    ("Eliesunne", "NA1"),
    ("NickHowler", "Wolf"),
    ("perruche19", "NA1"),
    ("fatbear07", "grrrr"),
    ("ukkari", "uwu"), # Iron
    
    # ///////////////////////////
    
    ("Harri", "00800"), # Iron
    ("kogollov", "NA1"), # Bronze
    ("Jeppy", "Nap"),
    ("Quon Tali", "EUW"),
    
    # ////////////////////////// 
    
    ("Scroop the Poop", "NA1"), # Emerald
    ("SCHIZOMODE", "CRZY"), # Plat
    ("Still Cold", "NA1"), # Gold
    ("bestplayerever", "NA2"), # Silver
    ("ItsSoji", "NA1"), # Bronze
    ("dragonbladz76", "1622") # Iron
]

MATCH_IDS_PER_PLAYER = 20
MAX_NEW_MATCHES_PER_CYCLE = 5000
PATCH_PREFIX_ALLOWLIST = None


# Riot API helper
def riot_get(url: str, params: Optional[Dict[str, Any]] = None):
    while True:
        r = requests.get(url, headers=HEADERS, params=params)

        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", "1"))
            print(f"[RATE LIMITED] Sleeping for {retry_after} seconds...")
            time.sleep(retry_after)
            continue

        if 500 <= r.status_code < 600:
            print("[SERVER ERROR] Retrying in 2 seconds...")
            time.sleep(2)
            continue

        r.raise_for_status()
        return r.json()

# API Calls
def get_account_by_riot_id(game_name: str, tag_line: str):
    url = f"https://{MATCH_REGION}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
    return riot_get(url)


def get_match_ids_by_puuid(puuid: str, count: int = 20):
    url = f"https://{MATCH_REGION}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
    params = {"start": 0, "count": count}
    return riot_get(url, params=params)


def get_match(match_id: str):
    url = f"https://{MATCH_REGION}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    return riot_get(url)


def get_rank_by_puuid(puuid: str):
    url = f"https://{REGION}.api.riotgames.com/lol/league/v4/entries/by-puuid/{puuid}"
    return riot_get(url)


# Schema for saving database
SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS match_participants (
    match_id TEXT NOT NULL,
    puuid TEXT NOT NULL,
    queue_id INTEGER,
    game_creation INTEGER,
    game_duration INTEGER,
    game_version TEXT,
    team_id INTEGER,
    champion_id INTEGER,
    champion_name TEXT,
    team_position TEXT,
    individual_position TEXT,
    win INTEGER,
    kills INTEGER,
    deaths INTEGER,
    assists INTEGER,
    PRIMARY KEY (match_id, puuid)
);

CREATE INDEX IF NOT EXISTS idx_match_id ON match_participants(match_id);
CREATE INDEX IF NOT EXISTS idx_puuid ON match_participants(puuid);
"""


def db_connect(path: str):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.executescript(SCHEMA_SQL)
    return conn


# Parameters:
# conn - sqlite3 connection
# match - dict from RIOT API /lol/match/v5/matches/{matchId}
def ingest_match(conn, match):
    metadata = match.get("metadata", {})
    info = match.get("info", {})

    match_id = metadata.get("matchId")
    if not match_id:
        return None

    if info.get("queueId") != QUEUE_ID_RANKED_SOLO:
        return None

    new_rows = 0

    for p in info.get("participants", []):
        cursor = conn.execute("""
            INSERT OR IGNORE INTO match_participants (
                match_id,
                puuid,
                queue_id,
                game_creation,
                game_duration,
                game_version,
                team_id,
                champion_id,
                champion_name,
                team_position,
                individual_position,
                win,
                kills,
                deaths,
                assists
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            match_id,
            p.get("puuid"),
            info.get("queueId"),
            info.get("gameCreation"),
            info.get("gameDuration"),
            info.get("gameVersion"),
            p.get("teamId"),
            p.get("championId"),
            p.get("championName"),
            p.get("teamPosition"),
            p.get("individualPosition"),
            1 if p.get("win") else 0,
            p.get("kills"),
            p.get("deaths"),
            p.get("assists"),
        ))

        if cursor.rowcount == 1:
            new_rows += 1

    return new_rows > 0



def api_cycle(conn):

    stats = {"new_matches": 0}

    player_queue = []

    for game_name, tag_line in SEED_RIOT_IDS:
        acct = get_account_by_riot_id(game_name, tag_line)
        player_queue.append(acct["puuid"])

    seen = set()

    while player_queue and stats["new_matches"] < MAX_NEW_MATCHES_PER_CYCLE:

        puuid = player_queue.pop(0)
        if puuid in seen:
            continue
        seen.add(puuid)

        now = int(time.time())

        # TODO: Move this to separate script
        
        # try:
        #     rank_data = get_rank_by_puuid(puuid)
        #     insert_rank_snapshot(conn, puuid, rank_data)
        # except Exception:
        #     pass

        match_ids = get_match_ids_by_puuid(puuid)

        for mid in match_ids:

            match = get_match(mid)

            with conn:
                ingested = ingest_match(conn, match)

            if ingested:
                stats["new_matches"] += 1

                for p in match["info"]["participants"]:
                    player_queue.append(p["puuid"])

    return stats


def main():
    conn = db_connect(DB_PATH)

    while True:
        print("\n--- api cycle start ---")
        stats = api_cycle(conn)
        print("Cycle stats:", stats)
        print("Sleeping for a minute..")
        time.sleep(60)


if __name__ == "__main__":
    main()
