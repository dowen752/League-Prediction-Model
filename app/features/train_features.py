import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# maps

tier_map = {
    "IRON": 0,
    "BRONZE": 4,
    "SILVER": 8,
    "GOLD": 12,
    "PLATINUM": 16,
    "EMERALD": 20,
    "DIAMOND": 24,
    "MASTER": 28,
    "GRANDMASTER": 32,
    "CHALLENGER": 36,
}

division_map = {
    "I": 0,
    "II": 1,
    "III": 2,
    "IV": 3
}

def build_train_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['champion_id'] = df['champion_id'].astype(str)
    df["team_position"] = df["team_position"].astype(str)

    df = df.sort_values(["puuid", "game_creation"]).copy()

    # Rolling winrate (NO leakage)
    df["win_shifted"] = df.groupby("puuid")["win"].shift(1)

    df["winrate"] = (
        df.groupby("puuid")["win_shifted"]
        .rolling(10, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["winrate"] = df["winrate"].fillna(0.5)
    df.drop(columns="win_shifted", inplace=True)

    # Rank encoding
    df["rank"] = df["tier"].map(tier_map) + df["division"].map(division_map)

    # team features

    def add_team_feature(df, col, prefix):
        team_vals = df.groupby(["match_id", "team_id"])[col].mean().unstack()
        df = df.merge(team_vals, on="match_id")

        df[f"team_avg_{prefix}"] = np.where(df["team_id"] == 100, df[100], df[200])
        df[f"enemy_avg_{prefix}"] = np.where(df["team_id"] == 100, df[200], df[100])

        df.drop(columns=[100, 200], inplace=True)
        return df

    df = add_team_feature(df, "league_points", "points")
    df["lp_diff"] = df["team_avg_points"] - df["enemy_avg_points"]

    df = add_team_feature(df, "rank", "rank")
    df = add_team_feature(df, "winrate", "wr")

    df["rank_diff"] = df["team_avg_rank"] - df["enemy_avg_rank"]
    df["winrate_diff"] = df["team_avg_wr"] - df["enemy_avg_wr"]

    return df


def split_train_test(df: pd.DataFrame):

    df = df[[
        'champion_id', 'team_position','league_points',
        'wins','losses','winrate',
        'rank','team_avg_points','enemy_avg_points',
        'lp_diff','win','match_id',
        'rank_diff','winrate_diff','team_avg_rank',
        'enemy_avg_rank','team_avg_wr','enemy_avg_wr',
        'team_id'
    ]]

    match_ids = df["match_id"].unique()

    train_ids, test_ids = train_test_split(
        match_ids, test_size=0.15, random_state=42
    )

    df_train = df[df["match_id"].isin(train_ids)].copy()
    df_test  = df[df["match_id"].isin(test_ids)].copy()

    # champion WR from train
    champ_wr = df_train.groupby("champion_id")["win"].mean()

    for d in [df_train, df_test]:
        d["champ_wr"] = d["champion_id"].map(champ_wr).fillna(0.5)

    # Team champion WR
    def add_cwr(df):
        team_vals = df.groupby(["match_id", "team_id"])["champ_wr"].mean().unstack()
        df = df.merge(team_vals, on="match_id")

        df["team_avg_cwr"] = np.where(df["team_id"] == 100, df[100], df[200])
        df["enemy_avg_cwr"] = np.where(df["team_id"] == 100, df[200], df[100])

        df.drop(columns=[100,200], inplace=True)

        df["champ_wr_diff"] = df["team_avg_cwr"] - df["enemy_avg_cwr"]

        return df

    df_train = add_cwr(df_train)
    df_test = add_cwr(df_test)

    # Cleanup
    df_train = df_train.drop(columns=["match_id","team_id"])
    df_test  = df_test.drop(columns=["match_id","team_id"])

    y_train = df_train["win"].replace({1: "Win", 0: "Loss"})
    y_test  = df_test["win"].replace({1: "Win", 0: "Loss"})

    X_train = df_train.drop(columns="win")
    X_test  = df_test.drop(columns="win")

    return X_train, X_test, y_train, y_test, champ_wr