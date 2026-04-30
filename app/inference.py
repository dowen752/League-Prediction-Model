import pandas as pd
import joblib
import requests
import time
from pathlib import Path
import os

from app.features.inference_features import build_inference_features


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "app" / "model" / "model.pkl"
META_PATH = BASE_DIR / "app" / "model" / "model_meta.pkl"
CHAMP_WR_PATH = BASE_DIR / "app" / "model" / "champ_wr.pkl"


RIOT_API_KEY = os.getenv("RIOT_API_KEY").strip()
if not RIOT_API_KEY:
    raise SystemExit("Missing RIOT_API_KEY, env not recognized.")
HEADERS = {"X-Riot-Token": RIOT_API_KEY}

REGION = "na1"
MATCH_REGION = "americas"

SLEEP_TIME = 0.1  # your request


# Helpers

def riot_get(url):
    response = requests.get(url, headers=HEADERS)
    time.sleep(SLEEP_TIME)

    if response.status_code != 200:
        return None

    return response.json()


def get_account(game_name, tag_line):
    from urllib.parse import quote

    game_name_enc = quote(game_name)
    tag_line_enc = quote(tag_line)

    url = f"https://{MATCH_REGION}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name_enc}/{tag_line_enc}"

    print("REQUEST:", url)

    response = requests.get(url, headers=HEADERS)
    print("STATUS:", response.status_code)
    print("RESPONSE:", response.text)

    time.sleep(SLEEP_TIME)

    if response.status_code != 200:
        return None

    return response.json()


def get_live_game(puuid):
    url = f"https://{REGION}.api.riotgames.com/lol/spectator/v5/active-games/by-summoner/{puuid}"
    return riot_get(url)


def get_match_ids(puuid, count=10):
    url = f"https://{MATCH_REGION}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?count={count}"
    return riot_get(url)


def get_match(match_id):
    url = f"https://{MATCH_REGION}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    return riot_get(url)

def get_summoner_by_puuid(puuid):
    url = f"https://{REGION}.api.riotgames.com/lol/summoner/v4/summoners/by-puuid/{puuid}"
    return riot_get(url)

def get_rank(puuid):
    url = f"https://{REGION}.api.riotgames.com/lol/league/v4/entries/by-puuid/{puuid}"
    data = riot_get(url)

    if not data:
        return None

    for entry in data:
        if entry["queueType"] == "RANKED_SOLO_5x5":
            return entry

    return None

    if not data:
        return None

    # prioritize ranked solo queue
    for entry in data:
        if entry["queueType"] == "RANKED_SOLO_5x5":
            return entry

    return None


def compute_winrate(puuid):
    match_ids = get_match_ids(puuid, 10)

    if not match_ids:
        return 0.5

    wins = 0
    total = 0

    for match_id in match_ids:
        match = get_match(match_id)
        if not match:
            continue

        for p in match["info"]["participants"]:
            if p["puuid"] == puuid:
                wins += int(p["win"])
                total += 1
                break

    if total == 0:
        return 0.5

    return wins / total


# Main class

class Predictor:

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.meta = joblib.load(META_PATH)
        self.champ_wr = joblib.load(CHAMP_WR_PATH)

        self.feature_columns = self.meta["feature_columns"]

    def build_player_df(self, game):

        players = []

        for p in game["participants"]:
            puuid = p["puuid"]

            winrate = compute_winrate(puuid)

            rank_data = get_rank(puuid)

            pos = p.get("teamPosition")

            if not pos:
                pos = "UTILITY"

            if rank_data:
                tier = rank_data["tier"]
                division = rank_data["rank"]
                lp = rank_data["leaguePoints"]
                wins = rank_data["wins"]
                losses = rank_data["losses"]
            else:
                tier = "GOLD"
                division = "IV"
                lp = 50
                wins = 50
                losses = 50
            players.append({
                "match_id": game["gameId"],
                "puuid": puuid,
                "team_id": p["teamId"],
                "champion_id": str(p["championId"]),
                "team_position": pos,
                "tier": tier,
                "division": division,
                "league_points": lp,
                "wins": wins,
                "losses": losses,
                "player_winrate": winrate
            })

        return pd.DataFrame(players)

    def preprocess(self, game):

        df = self.build_player_df(game)

        df_full = build_inference_features(df, self.champ_wr)

        df_model = df_full.reindex(columns=self.feature_columns)

        return df_full, df_model

    def predict_live(self, game_name, tag_line):

        account = get_account(game_name, tag_line)
        if not account:
            return {"error": "Account not found"}

        # summoner = get_summoner_by_puuid(account["puuid"])
        # if not summoner:
        #     return {"error": "Summoner not found"}

        # print("SUMMONER RESPONSE:", summoner)

        game = get_live_game(account["puuid"])

        if not game:
            return {"error": "Player not in a live game"}

        print(game.keys())
        print(type(game.get("participants")))

        df_full, df_model = self.preprocess(game)

        probs = self.model.predict_proba(df_model)

        df_full["prob"] = probs[:, 1]

        team_100 = df_full[df_full["team_id"] == 100]["prob"].mean()
        team_200 = df_full[df_full["team_id"] == 200]["prob"].mean()


        return {
            "blue team": float(team_100),
            "red team": float(team_200)
        }
        # return {
        #     "team_100_win_prob": float(team_100),
        #     "team_200_win_prob": float(team_200)
        # }