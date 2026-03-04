import os
import time
import sqlite3
from typing import Any, Dict, Optional
import requests
from dotenv import load_dotenv


load_dotenv()

RIOT_API_KEY = os.getenv("RIOT_API_KEY")
if not RIOT_API_KEY:
    raise SystemExit("Missing RIOT_API_KEY in environment.")

HEADERS = {"X-Riot-Token": RIOT_API_KEY}

REGION = "na1"
DB_PATH = "riot_lol.sqlite"

BATCH_SIZE = 100



def riot_get(url: str, params: Optional[Dict[str, Any]] = None):
    while True:
        r = requests.get(url, headers=HEADERS, params=params)

        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", "1"))
            print(f"[RATE LIMITED] Sleeping {retry_after}s...")
            time.sleep(retry_after)
            continue

        if 500 <= r.status_code < 600:
            print("[SERVER ERROR] Retrying in 2s...")
            time.sleep(2)
            continue

        r.raise_for_status()
        return r.json()


def get_rank_by_puuid(puuid: str):
    url = f"https://{REGION}.api.riotgames.com/lol/league/v4/entries/by-puuid/{puuid}"
    return riot_get(url)


# Helpers

def db_connect(path: str):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def get_all_players(conn):
    cursor = conn.execute("SELECT puuid FROM players;")
    return [row[0] for row in cursor.fetchall()]


def insert_rank_snapshot(conn, puuid, rank_data):
    now_ts = int(time.time())

    solo_entry = None
    for entry in rank_data:
        if entry["queueType"] == "RANKED_SOLO_5x5":
            solo_entry = entry
            break

    if solo_entry is None:
        return False  # unranked

    conn.execute("""
        INSERT OR IGNORE INTO player_ranks
        (puuid, tier, division, league_points, wins, losses, snapshot_ts)
        VALUES (?, ?, ?, ?, ?, ?, ?);
    """, (
        puuid,
        solo_entry.get("tier"),
        solo_entry.get("rank"),
        solo_entry.get("leaguePoints"),
        solo_entry.get("wins"),
        solo_entry.get("losses"),
        now_ts
    ))

    return True

def get_unranked_players(conn):
    cursor = conn.execute("""
        SELECT p.puuid
        FROM players p
        LEFT JOIN player_ranks r
            ON p.puuid = r.puuid
        WHERE r.puuid IS NULL;
    """)
    return [row[0] for row in cursor.fetchall()]


def fill_ranks():
    conn = db_connect(DB_PATH)
    players = get_unranked_players(conn)

    print(f"Found {len(players)} players to process.")

    processed = 0
    ranked = 0
    unranked = 0

    for puuid in players:
        try:
            rank_data = get_rank_by_puuid(puuid)

            with conn:
                success = insert_rank_snapshot(conn, puuid, rank_data)

            if success:
                ranked += 1
            else:
                unranked += 1

        except Exception as e:
            print(f"[ERROR] {puuid}: {e}")

        processed += 1

        if processed % BATCH_SIZE == 0:
            print(f"Processed {processed}/{len(players)}")

    print("\n=== DONE ===")
    print(f"Processed: {processed}")
    print(f"Ranked: {ranked}")
    print(f"Unranked: {unranked}")

    conn.close()


if __name__ == "__main__":
    fill_ranks()