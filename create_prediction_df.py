#%%
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


#print(build_data("Weibo Gaming", "Oh My God").ts())
# %%
