#%%
import joblib
import numpy as np
import pandas as pd

pd.DataFrame.ts = pd.DataFrame.to_string

player_data = pd.read_csv("players_export.csv")
player_data = player_data.drop(columns="Unnamed: 0")

blue_teams = player_data.filter(regex='blue_')
red_teams = player_data.filter(regex="red_")

blue_teams = blue_teams.iloc[::-1]
red_teams = red_teams.iloc[::-1]


test_data = pd.read_csv("predict_train.csv")
test_data = test_data.drop(columns=["Unnamed: 0", "result"])
#print(test_data.head(1).ts())

def build_data_row(blue_team, red_team):
    df = pd.DataFrame()
    blue_team_found = False
    red_team_found = False
    for blue_teamname, team in blue_teams.groupby("blue_teamname"):
        blue_name = team["blue_teamname"].values[0]
        if blue_team == blue_name:
            blue_team_row = team.head(1)
            blue_team_found = True
            break
    for red_teamname, team in red_teams.groupby("red_teamname"):
        red_name = team["red_teamname"].values[0]
        if red_team == red_name:
            red_team_row = team.head(1)
            red_team_found = True
            break
    if not blue_team_found or not red_team_found:
        raise KeyError("Team not in database")

    blue_elo_col = blue_team_row.pop("blue_elo")
    red_elo_col = red_team_row.pop("red_elo")

    blue_team_row.index = [0]
    red_team_row.index = [0]

    match_row = pd.concat([blue_team_row, red_team_row], axis=1)
    match_row = match_row.drop(columns=["blue_teamname", "red_teamname"])

    match_row["blue_elo"] = blue_elo_col.values[0]
    match_row["red_elo"] = red_elo_col.values[0]
    match_row["elo_diff"] = match_row["blue_elo"].values[0] - match_row["red_elo"].values[0]

    match_row["top_elo_diff"] = match_row["blue_top_elo"].values[0] - match_row["red_top_elo"].values[0]
    match_row["jng_elo_diff"] = match_row["blue_jng_elo"].values[0] - match_row["red_jng_elo"].values[0]
    match_row["mid_elo_diff"] = match_row["blue_mid_elo"].values[0] - match_row["red_mid_elo"].values[0]
    match_row["bot_elo_diff"] = match_row["blue_bot_elo"].values[0] - match_row["red_bot_elo"].values[0]
    match_row["sup_elo_diff"] = match_row["blue_sup_elo"].values[0] - match_row["red_sup_elo"].values[0]
    return match_row

def build_data(blue_team, red_team):
    blue_row = build_data_row(blue_team, red_team)
    red_row = build_data_row(red_team, blue_team)
    return pd.concat([blue_row, red_row])


def predict(blue_team, red_team):
    match_row = build_data(blue_team, red_team)
    encoded_data = pd.get_dummies(match_row, dtype=int)

    rf = joblib.load("Prediction-Model.job")
    useful_features = pd.read_csv("useful_features.csv")["feature"].tolist()

    X = encoded_data
    X_predict = X[useful_features]

    prediction = rf.predict(X_predict)
    probability = rf.predict_proba(X_predict)

    series_wins = 0
    blue_side = True

    for _ in range(100000):

        score, opp_score = 0, 0
        game_number = 1
        
        while score < 3 and opp_score < 3:
            blue_side = not blue_side
            if blue_side:
                p_blue_win = probability[0][1]
                if np.random.rand() < p_blue_win:
                    score += 1
                else:
                    opp_score += 1
            else:
                p_blue_win = probability[1][1]
                if np.random.rand() < p_blue_win:
                    opp_score += 1
                else:
                    score += 1
            game_number += 1
        if score == 3:
            series_wins += 1
        
    final_score = round(series_wins / 1000, 1)

    if final_score >= 50:
        return f"{blue_team} will win with {final_score}% confidence"
    else:
        return f"{red_team} will win with {100 - final_score}% confidence"

#%%
