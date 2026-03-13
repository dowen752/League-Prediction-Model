import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

# ///

# TODOs

# Might want to try difference btwn numeric vs categorical impact for rank

# get champion winrate out of our sampled games via champion_id aggregation, select win, take avg to get percent

# ///


def build_features(df: pd.DataFrame, gb = False) -> pd.DataFrame:
    df["winrate"] = df["wins"] / (df["wins"] + df["losses"])
    # roll into groups of league point values and take means
    match_team_points = df.groupby(["match_id", "team_id"])["league_points"]
    team_avgs = match_team_points.mean().unstack()
    # Include columns for match_ids 100 and 200
    df = df.merge(team_avgs, on="match_id")
    # print(team_avgs.head(5))

    # If players team_id is equal to 100, use df[100] otherwise use df[200] as their team average
    # And vice versa. essentially just selecting which column to fill with, depending on what team
    # assignment a player in df has
    df["team_avg_points"] = np.where(df["team_id"] == 100, df[100], df[200])
    df["enemy_avg_points"] = np.where(df["team_id"] == 100, df[200], df[100])

    # difference btwn team points
    df["lp_diff"] = df["team_avg_points"] - df["enemy_avg_points"]

    # Switching to categorical target
    df["win"] = df["win"].replace({1: "Win", 0: "Loss"})
    
    df["rank"] = df["tier"].map(tier_map) + df["division"].map(division_map) # encoded tier + div value between 0-40

    if gb:
        df["champion_id"] = df["champion_id"].astype(str)
    
    return df


def encoding_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df["team_position"] = df["team_position"].map(position_map) # Encoded val 0-4
    # df["rank"] = df["tier"].map(tier_map) + df["division"].map(division_map) # encoded tier + div value between 0-40
    return df



def train_test_val_sets(df, gb = False):
    if gb:
        X = df[['champion_id', 'team_position','league_points', 
                'wins', 'losses', 'winrate', 
                'rank', 'team_avg_points', 'enemy_avg_points', 
                'lp_diff']]
    else:
        X = df[['champion_id', 'team_position',
                'league_points', 'wins', 
                'losses', 'winrate', 
                'rank', 'team_avg_points', 
                'enemy_avg_points', 'lp_diff']].values
        
    y = df['win'].values

    # X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    # X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size = 0.5, random_state = 42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 42)

    return X_train, X_test, y_train, y_test
    

def random_forest_train(X_train, X_test, y_train, y_test, estimators):
    Random_Forest = RandomForestClassifier(n_estimators = estimators, random_state = 42)
    Random_Forest.fit(X_train, y_train)
    
    y_pred = Random_Forest.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Random Forest Accuracy at {estimators} estimators: {accuracy:.2f}")

def cat_boost_train(X_train, X_test, y_train, y_test, iters, lr = 0.05):
    cat_boost = CatBoostClassifier(
        iterations = iters,
        learning_rate = lr,
        random_seed = 42
    )

    cat_boost.fit(X_train, y_train, cat_features = ["team_position", "champion_id"], verbose = False)
    y_pred =cat_boost.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Cat Boost Accuracy at {iters} iterations, lr = {lr:.2f} : {accuracy:.2f}")



position_map = {
    "TOP" : 0,
    "JUNGLE" : 1,
    "MIDDLE" : 2, 
    "BOTTOM" : 3,
    "UTILITY" : 4
}
    
tier_map = {
    "IRON" : 0,
    "BRONZE" : 4,
    "SILVER" : 8,
    "GOLD": 12,
    "PLATINUM" : 16,
    "EMERALD" : 20,
    "DIAMOND" : 24,
    "MASTER" : 28,
    "GRANDMASTER" : 32,
    "CHALLENGER" : 36,
}

division_map = {
    "I" : 0,
    "II" : 1,
    "III" : 2,
    "IV" : 3
}
