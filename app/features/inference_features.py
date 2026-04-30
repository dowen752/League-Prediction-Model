import pandas as pd
import numpy as np

from .train_features import tier_map, division_map

def build_inference_features(df: pd.DataFrame, champ_wr_map: dict):

    # need:
    # - puuid
    # - team_id
    # - champion_id
    # - tier, division
    # - league_points
    # - wins, losses
    # - player_winrate  (computed from Riot API)

    df = df.copy()

    df["champion_id"] = df["champion_id"].astype(str)
    df["team_position"] = df["team_position"].astype(str)

    # using provided winrate
    df["winrate"] = df["player_winrate"].fillna(0.5)

    # rank encoding
    df["rank"] = df["tier"].map(tier_map) + df["division"].map(division_map)

    # champion WR
    df["champ_wr"] = df["champion_id"].map(champ_wr_map).fillna(0.5)

    # team features

    def add_team_feature(df, col, prefix):
        team_vals = (
            df.groupby(["match_id", "team_id"])[col]
            .mean()
            .reset_index()
        )
        team_vals = team_vals.pivot(index="match_id", columns="team_id", values=col)

        # merge back
        df = df.merge(team_vals, on="match_id", how="left")
        df[f"team_avg_{prefix}"] = np.where(df["team_id"] == 100, df[100], df[200])
        df[f"enemy_avg_{prefix}"] = np.where(df["team_id"] == 100, df[200], df[100])

        df = df.drop(columns=[100, 200], errors="ignore")

        return df

    df = add_team_feature(df, "league_points", "points")
    df["lp_diff"] = df["team_avg_points"] - df["enemy_avg_points"]

    df = add_team_feature(df, "rank", "rank")
    df = add_team_feature(df, "winrate", "wr")
    df = add_team_feature(df, "champ_wr", "cwr")

    df["rank_diff"] = df["team_avg_rank"] - df["enemy_avg_rank"]
    df["winrate_diff"] = df["team_avg_wr"] - df["enemy_avg_wr"]
    df["champ_wr_diff"] = df["team_avg_cwr"] - df["enemy_avg_cwr"]

    # Match training structure
    df = df[[
        'champion_id','team_position','league_points',
        'wins','losses','winrate',
        'rank','team_avg_points','enemy_avg_points',
        'lp_diff',
        'rank_diff','winrate_diff',
        'team_avg_rank','enemy_avg_rank',
        'team_avg_wr','enemy_avg_wr',
        'team_avg_cwr','enemy_avg_cwr',
        'champ_wr','champ_wr_diff',
        'team_id' # for last engineering, dropped later
    ]]

    return df
    