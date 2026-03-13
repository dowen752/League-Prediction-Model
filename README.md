Anyone who has played League of Legends knows the feeling of a game being “over” long before seeing the defeat screen. This project explores whether that outcome can be predicted before the match even loads in.
Using data collected directly from the Riot Games API, I built a full ML pipeline that:
- Crawls ranked match data at scale
- Stores relational player/match data in SQLite
- Engineers team-level features from rank, LP, and champion statistics
- Trains a classification model to predict pre-game win probability

The focus of this project is end-to-end system design: data ingestion, feature construction, and model evaluation.
