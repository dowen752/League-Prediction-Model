import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

# ///

# TODOs

# get champion winrate out of our sampled games via champion_id aggregation, select win, take avg to get percent

# ///


def build_features(df: pd.DataFrame, gb = False) -> pd.DataFrame:
    

    
    df = df.sort_values(["puuid", "game_creation"]).copy()
    df["win_shifted"] = df.groupby("puuid")["win"].shift(1) # removing current from wr calculation
    df["winrate"] = (
        df.groupby("puuid")["win_shifted"]
        .rolling(10, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["winrate"] = df["winrate"].fillna(0.5)
    df.drop(columns="win_shifted", inplace = True)

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
    df.drop(columns=[100,200], inplace = True) # resetting 100 200 columns

    # difference btwn team points
    df["lp_diff"] = df["team_avg_points"] - df["enemy_avg_points"]

    # rank mapping from division and tier -> single numeric
    df["rank"] = df["tier"].map(tier_map) + df["division"].map(division_map) # encoded tier + div value between 0-40

    
    match_team_rank = df.groupby(["match_id", "team_id"])["rank"]
    team_rank_avgs = match_team_rank.mean().unstack()
    
    df = df.merge(team_rank_avgs, on="match_id")
    
    df["team_avg_rank"] = np.where(df["team_id"] == 100, df[100], df[200])
    df["enemy_avg_rank"] = np.where(df["team_id"] == 100, df[200], df[100])
    df.drop(columns=[100,200], inplace = True) # resetting 100 200 columns

    
    match_team_wr = df.groupby(["match_id", "team_id"])["winrate"]
    team_wr_avgs = match_team_wr.mean().unstack()
    
    df = df.merge(team_wr_avgs, on="match_id")
    
    df["team_avg_wr"] = np.where(df["team_id"] == 100, df[100], df[200])
    df["enemy_avg_wr"] = np.where(df["team_id"] == 100, df[200], df[100])
    df.drop(columns=[100,200], inplace = True) # resetting 100 200 columns

    df["rank_diff"] = df['team_avg_rank'] - df['enemy_avg_rank']
    df["winrate_diff"] = df['team_avg_wr'] - df['enemy_avg_wr']

    if gb:
        df["champion_id"] = df["champion_id"].astype(str)
    
    return df


def encoding_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df["team_position"] = df["team_position"].map(position_map) # Encoded val 0-4
    # df["rank"] = df["tier"].map(tier_map) + df["division"].map(division_map) # encoded tier + div value between 0-40
    return df



def train_test_val_sets(df: pd.DataFrame, gb = False):
    
    df = df[['champion_id', 'team_position','league_points', 
                'wins', 'losses', 'winrate', 
                'rank', 'team_avg_points', 'enemy_avg_points', 
                'lp_diff', 'win', 'match_id',
                'rank_diff', 'winrate_diff','team_avg_rank', 
                'enemy_avg_rank', 'team_avg_wr','enemy_avg_wr',
                'team_id']]

    match_ids = df["match_id"].unique()
    
    train_ids, test_ids = train_test_split(match_ids, test_size=0.15, random_state=42)
    
    df_train = df[df["match_id"].isin(train_ids)]
    df_test  = df[df["match_id"].isin(test_ids)]
    
    
    df_train = df_train.copy()
    df_test = df_test.copy()
    
    wr_train = df_train.groupby("champion_id")["win"].mean()
    
    df_train["champ_wr"] = df_train["champion_id"].map(wr_train)
    df_test["champ_wr"] = df_test["champion_id"].map(wr_train)
    
    df_train["champ_wr"] = df_train["champ_wr"].fillna(0.5)
    df_test["champ_wr"] = df_test["champ_wr"].fillna(0.5)

    # /////////////////////////////////////////////////////////////////////
    match_team_cwr_train = df_train.groupby(["match_id", "team_id"])["champ_wr"]
    team_cwr_avgs_train = match_team_cwr_train.mean().unstack()
    
    df_train = df_train.merge(team_cwr_avgs_train, on="match_id")
    
    df_train["team_avg_cwr"] = np.where(df_train["team_id"] == 100, df_train[100], df_train[200])
    df_train["enemy_avg_cwr"] = np.where(df_train["team_id"] == 100, df_train[200], df_train[100])
    df_train.drop(columns=[100,200], inplace = True) # resetting 100 200 columns
    # /////////////////////////////////////////////////////////////////////


    
    # /////////////////////////////////////////////////////////////////////
    match_team_cwr_test = df_test.groupby(["match_id", "team_id"])["champ_wr"]
    team_cwr_avgs_test = match_team_cwr_test.mean().unstack()
    
    df_test = df_test.merge(team_cwr_avgs_test, on="match_id")
    
    df_test["team_avg_cwr"] = np.where(df_test["team_id"] == 100, df_test[100], df_test[200])
    df_test["enemy_avg_cwr"] = np.where(df_test["team_id"] == 100, df_test[200], df_test[100])
    df_test.drop(columns=[100,200], inplace = True) # resetting 100 200 columns
    # /////////////////////////////////////////////////////////////////////

    df_train["champ_wr_diff"] = df_train['team_avg_cwr'] - df_train['enemy_avg_cwr']
    df_test["champ_wr_diff"] = df_test['team_avg_cwr'] - df_test['enemy_avg_cwr']


    df_train = df_train.drop(columns = ['match_id', 'team_id'])
    df_test = df_test.drop(columns = ['match_id', 'team_id'])

    
    # Switching to categorical targets
    y_train = df_train['win'].replace({1: "Win", 0: "Loss"})
    y_test = df_test['win'].replace({1: "Win", 0: "Loss"})

    X_train = df_train.drop(columns = 'win')
    X_test = df_test.drop(columns = 'win')
    
    return X_train, X_test, y_train, y_test
    

def random_forest_train(X_train, X_test, y_train, y_test, estimators):
    Random_Forest = RandomForestClassifier(n_estimators = estimators, random_state = 42)
    Random_Forest.fit(X_train, y_train)
    
    y_pred = Random_Forest.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Random Forest Accuracy at {estimators} estimators: {accuracy:.2f}")



def feature_selection(X_train, X_test, y_train, y_test, iters = 1000, lr = 0.1):
    current_features = list(X_train.columns)
    
    best_score = cat_boost_train(X_train, X_test, y_train, y_test, iters, lr)
    print(f"Initial score: {best_score:.4f}")
    
    improved = True
    
    while improved:
        improved = False
        best_feature = None
        best_new_score = best_score
        
        for col in current_features:
            trial_features = [f for f in current_features if f != col]
            
            X_train_trial = X_train[trial_features]
            X_test_trial = X_test[trial_features]
            
            score = cat_boost_train(X_train_trial, X_test_trial, y_train, y_test, iters, lr)
            
            print(f"Test removing {col}: {score:.4f}")
            
            if score > best_new_score:
                best_new_score = score
                best_feature = col
        
        if best_feature is not None:
            print(f"\nRemoving {best_feature} improved score: {best_new_score:.4f}\n")
            current_features.remove(best_feature)
            best_score = best_new_score
            improved = True
    
    print("\nFinal feature set:", current_features)
    print("Final score:", best_score)
    
    return current_features

def cat_boost_train(X_train, X_test, y_train, y_test, iters = 1000, learn_rate = 0.1):

    X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42
    )

   # model = CatBoostClassifier(iterations=iters, learning_rate=learn_rate, verbose=True)

    model = CatBoostClassifier(
    iterations=iters,
    learning_rate=0.03,
    depth=8,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=42,
    verbose=100
    )

    model.fit(
    X_tr, y_tr,
    cat_features=get_cat_features(X_tr),
    eval_set=(X_val, y_val),
    early_stopping_rounds=200,
    use_best_model=True
    )


    # //////////////////////// TESTING DATA LEAKAGE ////////////////////////

    # y_train = np.random.permutation(y_train)

    # //////////////////////// TESTING DATA LEAKAGE ////////////////////////

    
    # model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=get_cat_features(X_train))

    # //////////////////////// TESTING OVERFITTING ////////////////////////

    # train_preds = model.predict(X_train)
    # test_preds = model.predict(X_test)
    
    # train_acc = accuracy_score(y_train, train_preds)
    # test_acc = accuracy_score(y_test, test_preds)

    # print(train_acc, test_acc)

    # //////////////////////// TESTING OVERFITTING ////////////////////////
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    return acc

def get_cat_features(X):
    cat_cols = ["champion_id", "team_position"]
    return [col for col in cat_cols if col in X.columns]


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
