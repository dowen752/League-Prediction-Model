import pandas as pd
import sqlite3
import joblib
from pathlib import Path

from catboost import CatBoostClassifier
from app.features.train_features import build_train_features, split_train_test

BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = BASE_DIR / "data" / "riot_lol.sqlite"
MODEL_PATH = BASE_DIR / "app" / "model" / "model.pkl"
META_PATH = BASE_DIR / "app" / "model" / "model_meta.pkl"
CHAMP_WR_PATH = BASE_DIR / "app" / "model" / "champ_wr.pkl"



def load_data():
    conn = sqlite3.connect(DB_PATH)

    query = """
    SELECT 
        p.match_id,
        p.team_id,
        p.puuid,
        p.win,
        p.champion_id,
        p.team_position,
        r.tier,
        r.division,
        r.league_points,
        r.wins,
        r.losses,
        m.game_creation
    FROM participants p
    LEFT JOIN matches m
        ON p.match_id = m.match_id
    LEFT JOIN player_ranks r
        ON r.puuid = p.puuid
        AND r.snapshot_ts = (
            SELECT MAX(r2.snapshot_ts)
            FROM player_ranks r2
            WHERE r2.puuid = p.puuid
            AND r2.snapshot_ts <= m.game_creation
        )
    """

    df = pd.read_sql(query, conn)
    conn.close()
    return df



def train():
    print("Loading data...")
    df = load_data()

    print("Building training features...")
    df = build_train_features(df)

    print("Splitting data...")
    X_train, X_test, y_train, y_test, champ_wr = split_train_test(df)

    print("Training model...")
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=8,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=42,
        verbose=100
    )

    model.fit(
        X_train,
        y_train,
        cat_features=["champion_id", "team_position"],
        eval_set=(X_test, y_test),
        early_stopping_rounds=200,
        use_best_model=True
    )

    print("Saving model...")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    # Metadata, champ wr constants

    meta = {
        "feature_columns": list(X_train.columns),
        "cat_features": ["champion_id", "team_position"]
    }

    joblib.dump(meta, META_PATH)
    joblib.dump(champ_wr.to_dict(), CHAMP_WR_PATH)

    print("Done.")


if __name__ == "__main__":
    train()