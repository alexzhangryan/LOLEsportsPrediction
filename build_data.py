# %%
from cycler import V
from matplotlib import pyplot as plt
import pandas as pd
import textwrap
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.metrics import classification_report
import xgboost as xgb


df = pd.read_csv("match_data.csv")
df = df.drop(columns=["datacompleteness", "url", "participantid", "playerid", "teamid"])
df = df.fillna({"playername": "none"})
teams = df[df["playername"] == "none"]
teams = teams.drop(columns=["doublekills", "triplekills", "quadrakills", "pentakills", "firstbloodkill", "firstbloodassist", "firstbloodvictim", "elementaldrakes", "opp_elementaldrakes", "infernals", "mountains", "clouds", "oceans", "chemtechs", "hextechs", "dragons (type unknown)", "elders", "opp_elders", "firstherald", "atakhans", "opp_atakhans", "firstmidtower", "firsttothreetowers", "turretplates", "opp_turretplates", "damagemitigatedperminute", "damagetotowers", "gpr", "cspm", "damageshare", "earnedgoldshare", "total cs", "minionkills", "firstbaron", "champion", "playername", "position", "date", "year"])
teams = teams.iloc[:, :60]
teams2025 = teams[(teams["league"] == "LCK") | (teams["league"] == "LPL") | (teams["league"] == "LCK") | (teams["league"] == "LTA N") | (teams["league"] == "LTA S") | (teams["league"] == "LCP") | (teams["league"] == "LEC") | (teams["league"] == "EWC") | (teams["league"] == "FST") | (teams["league"] == "MSI") | (teams["league"] == "WLDs")]

#TEAMS

blue = teams2025[teams2025["side"] == "Blue"].copy()
red = teams2025[teams2025["side"] == "Red"].copy()
blue = blue.add_prefix("blue_")
red = red.add_prefix("red_")

blue = blue.rename(columns={"blue_gameid": "gameid"})
red = red.rename(columns={"red_gameid": "gameid"})


blue_ids = set(blue["gameid"])
red_ids = set(red["gameid"])

matches = pd.merge(blue, red, on="gameid", suffixes=("",""))
matches["blue_elo"] = 1500
matches["red_elo"] = 1500
region_strength2025 = {
    "LCK": 1755,
    "LPL": 1604,
    "LTA N": 1387,
    "LEC": 1341,
    "LCP": 1294,
    "LTA S": 824
}


#initialize players
players = df[df["playername"] != "none"]
players = players.drop(columns=[ "pick1", "pick2", "pick3", "pick4", "pick5", "doublekills", "triplekills", "quadrakills", "pentakills", "firstblood", "firstbloodassist", "firstbloodvictim","firstdragon", "dragons", "opp_dragons", "elementaldrakes", "opp_elementaldrakes", "infernals", "mountains", "clouds", "oceans", "chemtechs", "hextechs", "dragons (type unknown)", "elders", "opp_elders", "firstherald", "atakhans", "opp_atakhans", "firstmidtower", "firsttothreetowers", "turretplates", "opp_turretplates", "heralds", "opp_heralds", "void_grubs", "opp_void_grubs", "firstbaron", "barons", "opp_barons", "firsttower", "towers", "opp_towers", "inhibitors", "opp_inhibitors", "damagemitigatedperminute", "damagetotowers", "gspd", "gpr", "monsterkillsownjungle", "monsterkillsenemyjungle"])
players = players.iloc[:, :48]
players2025 = players[(players["league"] == "LCK") | (players["league"] == "LPL") | (players["league"] == "LCK") | (players["league"] == "LTA N") | (players["league"] == "LTA S") | (players["league"] == "LCP") | (players["league"] == "LEC") | (players["league"] == "EWC") | (players["league"] == "FST") | (players["league"] == "MSI") | (players["league"] == "WLDs")]

#PLAYER DATA
#%%
blue_players = players2025[players2025["side"] == "Blue"].copy()
red_players = players2025[players2025["side"] == "Red"].copy()

blue_players = blue_players.add_prefix("blue_")
red_players = red_players.add_prefix("red_")

blue_players = blue_players.drop(columns=["blue_league", "blue_year", "blue_split", "blue_playoffs", "blue_date", "blue_patch", "blue_game"])
red_players = red_players.drop(columns=["red_league", "red_year", "red_split", "red_playoffs", "red_date", "red_patch", "red_game"])

blue_players = blue_players.rename(columns={"blue_gameid": "gameid"})
red_players = red_players.rename(columns={"red_gameid": "gameid"})

blue_players_ids = set(blue_players["gameid"])
red_players_ids = set(red_players["gameid"])

blue_top = blue_players[blue_players["blue_position"] == "top"].copy()
blue_jng = blue_players[blue_players["blue_position"] == "jng"].copy()
blue_mid = blue_players[blue_players["blue_position"] == "mid"].copy()
blue_bot = blue_players[blue_players["blue_position"] == "bot"].copy()
blue_sup = blue_players[blue_players["blue_position"] == "sup"].copy()

red_top = red_players[red_players["red_position"] == "top"].copy()
red_jng = red_players[red_players["red_position"] == "jng"].copy()
red_mid = red_players[red_players["red_position"] == "mid"].copy()
red_bot = red_players[red_players["red_position"] == "bot"].copy()
red_sup = red_players[red_players["red_position"] == "sup"].copy()

blue_top = blue_top.add_suffix("_top")
blue_jng = blue_jng.add_suffix("_jng")
blue_mid = blue_mid.add_suffix("_mid")
blue_bot = blue_bot.add_suffix("_bot")
blue_sup = blue_sup.add_suffix("_sup")

red_top = red_top.add_suffix("_top")
red_jng = red_jng.add_suffix("_jng")
red_mid = red_mid.add_suffix("_mid")
red_bot = red_bot.add_suffix("_bot")
red_sup = red_sup.add_suffix("_sup")

blue_top = blue_top.rename(columns={"gameid_top": "gameid"})
blue_jng = blue_jng.rename(columns={"gameid_jng": "gameid"})
blue_mid = blue_mid.rename(columns={"gameid_mid": "gameid"})
blue_bot = blue_bot.rename(columns={"gameid_bot": "gameid"})
blue_sup = blue_sup.rename(columns={"gameid_sup": "gameid"})

red_top = red_top.rename(columns={"gameid_top": "gameid"})
red_jng = red_jng.rename(columns={"gameid_jng": "gameid"})
red_mid = red_mid.rename(columns={"gameid_mid": "gameid"})
red_bot = red_bot.rename(columns={"gameid_bot": "gameid"})
red_sup = red_sup.rename(columns={"gameid_sup": "gameid"})

player_matches = pd.merge(blue_top, blue_jng, on="gameid", suffixes=("",""))
player_matches = pd.merge(player_matches, blue_mid, on="gameid", suffixes=("",""))
player_matches = pd.merge(player_matches, blue_bot, on="gameid", suffixes=("",""))
player_matches = pd.merge(player_matches, blue_sup, on="gameid", suffixes=("",""))
player_matches = pd.merge(player_matches, red_top, on="gameid", suffixes=("",""))
player_matches = pd.merge(player_matches, red_jng, on="gameid", suffixes=("",""))
player_matches = pd.merge(player_matches, red_mid, on="gameid", suffixes=("",""))
player_matches = pd.merge(player_matches, red_bot, on="gameid", suffixes=("",""))
player_matches = pd.merge(player_matches, red_sup, on="gameid", suffixes=("",""))

players_match_metadata = players2025.iloc[:, 0:8]
players_match_metadata = players_match_metadata.drop_duplicates(subset="gameid")
player_matches = pd.merge(players_match_metadata, player_matches, on="gameid", suffixes=("",""))

#NO GAME DATA

player_averages = {}

for col in players2025.select_dtypes(include="number").columns:
    player_averages[col] = players2025.groupby("playername")[col].mean()

player_avgs = pd.DataFrame(player_averages)
player_avgs = player_avgs.drop(columns=["year", "playoffs", "game", "patch", "result"])
player_avgs = player_avgs.add_prefix("avg_")
player_avgs = player_avgs.reset_index()


players_no_game_data = players2025.iloc[:, 0:12]
players_no_game_data["result"] = players2025["result"].copy()
results_col = players_no_game_data.pop("result")
players_no_game_data.insert(0, "result", results_col)
#%%

blue_players = players_no_game_data[players_no_game_data["side"] == "Blue"].copy()
red_players = players_no_game_data[players_no_game_data["side"] == "Red"].copy()

blue_players = blue_players.drop(columns=["league", "year", "split", "playoffs", "date", "patch", "game"])
red_players = red_players.drop(columns=["league", "year", "split", "playoffs", "date", "patch", "game"])

blue_players_ids = set(blue_players["gameid"])
red_players_ids = set(red_players["gameid"])

blue_top = blue_players[blue_players["position"] == "top"].copy()
blue_jng = blue_players[blue_players["position"] == "jng"].copy()
blue_mid = blue_players[blue_players["position"] == "mid"].copy()
blue_bot = blue_players[blue_players["position"] == "bot"].copy()
blue_sup = blue_players[blue_players["position"] == "sup"].copy()

red_top = red_players[red_players["position"] == "top"].copy()
red_jng = red_players[red_players["position"] == "jng"].copy()
red_mid = red_players[red_players["position"] == "mid"].copy()
red_bot = red_players[red_players["position"] == "bot"].copy()
red_sup = red_players[red_players["position"] == "sup"].copy()

blue_top = pd.merge(blue_top, player_avgs, on="playername", suffixes=("",""))
blue_jng = pd.merge(blue_jng, player_avgs, on="playername", suffixes=("",""))
blue_mid = pd.merge(blue_mid, player_avgs, on="playername", suffixes=("",""))
blue_bot = pd.merge(blue_bot, player_avgs, on="playername", suffixes=("",""))
blue_sup = pd.merge(blue_sup, player_avgs, on="playername", suffixes=("",""))

red_top = pd.merge(red_top, player_avgs, on="playername", suffixes=("",""))
red_jng = pd.merge(red_jng, player_avgs, on="playername", suffixes=("",""))
red_mid = pd.merge(red_mid, player_avgs, on="playername", suffixes=("",""))
red_bot = pd.merge(red_bot, player_avgs, on="playername", suffixes=("",""))
red_sup = pd.merge(red_sup, player_avgs, on="playername", suffixes=("",""))

blue_top = blue_top.add_prefix("blue_top_")
blue_jng = blue_jng.add_prefix("blue_jng_")
blue_mid = blue_mid.add_prefix("blue_mid_")
blue_bot = blue_bot.add_prefix("blue_bot_")
blue_sup = blue_sup.add_prefix("blue_sup_")

red_top = red_top.add_prefix("red_top_")
red_jng = red_jng.add_prefix("red_jng_")
red_mid = red_mid.add_prefix("red_mid_")
red_bot = red_bot.add_prefix("red_bot_")
red_sup = red_sup.add_prefix("red_sup_")

#%%
blue_top = blue_top.drop(columns=["blue_top_side", "blue_top_position"])
blue_top = blue_top.rename(columns={"blue_top_teamname": "blue_teamname"})
blue_top = blue_top.rename(columns={"blue_top_result": "result"})
blue_top_col = blue_top.pop("blue_teamname")
blue_top.insert(1, "blue_teamname", blue_top_col)

blue_jng = blue_jng.drop(columns=["blue_jng_side", "blue_jng_position", "blue_jng_teamname", "blue_jng_result"])

blue_mid = blue_mid.drop(columns=["blue_mid_side", "blue_mid_position", "blue_mid_teamname", "blue_mid_result"])

blue_bot = blue_bot.drop(columns=["blue_bot_side", "blue_bot_position", "blue_bot_teamname", "blue_bot_result"])

blue_sup = blue_sup.drop(columns=["blue_sup_side", "blue_sup_position", "blue_sup_teamname", "blue_sup_result"])

red_top = red_top.drop(columns=["red_top_side", "red_top_position", "red_top_result"])
red_top = red_top.rename(columns={"red_top_teamname": "red_teamname"})
red_top_col = red_top.pop("red_teamname")
red_top.insert(1, "red_teamname", red_top_col)

red_jng = red_jng.drop(columns=["red_jng_side", "red_jng_position", "red_jng_teamname", "red_jng_result"])

red_mid = red_mid.drop(columns=["red_mid_side", "red_mid_position", "red_mid_teamname", "red_mid_result"])

red_bot = red_bot.drop(columns=["red_bot_side", "red_bot_position", "red_bot_teamname", "red_bot_result"])

red_sup = red_sup.drop(columns=["red_sup_side", "red_sup_position", "red_sup_teamname", "red_sup_result"])

blue_top = blue_top.rename(columns={"blue_top_gameid": "gameid"})
blue_jng = blue_jng.rename(columns={"blue_jng_gameid": "gameid"})
blue_mid = blue_mid.rename(columns={"blue_mid_gameid": "gameid"})
blue_bot = blue_bot.rename(columns={"blue_bot_gameid": "gameid"})
blue_sup = blue_sup.rename(columns={"blue_sup_gameid": "gameid"})

red_top = red_top.rename(columns={"red_top_gameid": "gameid"})
red_jng = red_jng.rename(columns={"red_jng_gameid": "gameid"})
red_mid = red_mid.rename(columns={"red_mid_gameid": "gameid"})
red_bot = red_bot.rename(columns={"red_bot_gameid": "gameid"})
red_sup = red_sup.rename(columns={"red_sup_gameid": "gameid"})

#%%

team_elo = {}
player_elo = {}
team_region = {}
player_region = {}
player_blue_winrate = {}
player_red_winrate = {}
#%%
def expectedScore(rating_a, rating_b):
    return (1/(1 + 10**((rating_b-rating_a)/400)))

def updateElo(rating_a, rating_b, score_a, weight, K):
    expected_a = expectedScore(rating_a, rating_b)
    expected_b = 1 - expected_a

    new_rating_a = int(rating_a + K * weight * (score_a - expected_a))
    new_rating_b = int(rating_b + K * weight * ((1 - score_a) - expected_b))

    return [new_rating_a, new_rating_b]

def initializeTeamElo(matches):
    for gameid, match in matches.groupby("gameid"):
        blue_team = match["blue_teamname"].values[0]
        red_team = match["red_teamname"].values[0]

        rating_a = team_elo.get(blue_team, 1500)
        rating_b = team_elo.get(red_team, 1500)

        region_a = region_strength2025.get(team_region.get(blue_team, "LEC"))
        region_b = region_strength2025.get(team_region.get(red_team, "LEC"))
        weight = (region_a / region_b) ** 0.5

        score_a = match.iloc[0]["blue_result"]
        try:
            elos = updateElo(rating_a, rating_b, score_a, weight, 30)
        except:
            pass
        team_elo[blue_team] = elos[0]
        team_elo[red_team] = elos[1]
    for gameid, match in matches.groupby("gameid"):
        blue_team = match["blue_teamname"].values[0]
        red_team = match["red_teamname"].values[0]
        matches.loc[match.index, "blue_elo"] = team_elo.get(blue_team)
        matches.loc[match.index, "red_elo"] = team_elo.get(red_team)

        
        

#%%
def initializeTeams(teams):
    for gameid, team in teams.groupby("gameid"):
        team_elo[team["teamname"].values[0]] = 1500
        if team["league"].values[0] != "FST" and team["league"].values[0] != "EWC" and team["league"].values[0] != "MSI" and team["league"].values[0] != "WLDs":
            team_region[team["teamname"].values[0]] = team["league"].values[0]
    initializeTeamElo(matches)

initializeTeams(teams2025)
#%%

def initializePlayers(players):
    
    for gameid, player in players.groupby("gameid"):
        blue_top = player["blue_playername_top"].values[0]
        player_elo[blue_top] = 1500
        blue_jng = player["blue_playername_jng"].values[0]
        player_elo[blue_jng] = 1500
        blue_mid = player["blue_playername_mid"].values[0]
        player_elo[blue_mid] = 1500
        blue_bot = player["blue_playername_bot"].values[0]
        player_elo[blue_bot] = 1500
        blue_sup = player["blue_playername_sup"].values[0]
        player_elo[blue_sup] = 1500

        red_top = player["red_playername_top"].values[0]
        player_elo[red_top] = 1500
        red_jng = player["red_playername_jng"].values[0]
        player_elo[red_jng] = 1500
        red_mid = player["red_playername_mid"].values[0]
        player_elo[red_mid] = 1500
        red_bot = player["red_playername_bot"].values[0]
        player_elo[red_bot] = 1500
        red_sup = player["red_playername_sup"].values[0]
        player_elo[red_sup] = 1500

        league = player["league"].values[0]
        if league != "FST" and league != "EWC" and league != "MSI" and league != "WLDs":
            player_region[blue_top] = player["league"].values[0]
            player_region[blue_jng] = player["league"].values[0]
            player_region[blue_mid] = player["league"].values[0]
            player_region[blue_bot] = player["league"].values[0]
            player_region[blue_sup] = player["league"].values[0]

            player_region[red_top] = player["league"].values[0]
            player_region[red_jng] = player["league"].values[0]
            player_region[red_mid] = player["league"].values[0]
            player_region[red_bot] = player["league"].values[0]
            player_region[red_sup] = player["league"].values[0]

    for gameid, player in players.groupby("gameid"):
        #top
        blue_top = player["blue_playername_top"].values[0]
        red_top = player["red_playername_top"].values[0]

        rating_a = player_elo.get(blue_top, 1500)
        rating_b = player_elo.get(red_top, 1500)

        region_a = region_strength2025.get(player_region.get(blue_top, "LEC"))
        region_b = region_strength2025.get(player_region.get(red_top, "LEC"))
        weight = (region_a / region_b) ** 0.5

        score_a = player.iloc[0]["blue_result_top"]
        try:
            elos = updateElo(rating_a, rating_b, score_a, weight, 30)
        except:
            pass
        player_elo[blue_top] = elos[0]
        player_elo[red_top] = elos[1]


        #jungle

        blue_jng = player["blue_playername_jng"].values[0]
        red_jng = player["red_playername_jng"].values[0]

        rating_a = player_elo.get(blue_jng, 1500)
        rating_b = player_elo.get(red_jng, 1500)

        region_a = region_strength2025.get(player_region.get(blue_jng, "LEC"))
        region_b = region_strength2025.get(player_region.get(red_jng, "LEC"))
        weight = (region_a / region_b) ** 0.5

        score_a = player.iloc[0]["blue_result_jng"]
        try:
            elos = updateElo(rating_a, rating_b, score_a, weight, 30)
        except:
            pass
        player_elo[blue_jng] = elos[0]
        player_elo[red_jng] = elos[1]

        #mid

        blue_mid = player["blue_playername_mid"].values[0]
        red_mid = player["red_playername_mid"].values[0]

        rating_a = player_elo.get(blue_mid, 1500)
        rating_b = player_elo.get(red_mid, 1500)

        region_a = region_strength2025.get(player_region.get(blue_mid, "LEC"))
        region_b = region_strength2025.get(player_region.get(red_mid, "LEC"))
        weight = (region_a / region_b) ** 0.5

        score_a = player.iloc[0]["blue_result_mid"]
        try:
            elos = updateElo(rating_a, rating_b, score_a, weight, 30)
        except:
            pass
        player_elo[blue_mid] = elos[0]
        player_elo[red_mid] = elos[1]

        #bot

        blue_bot = player["blue_playername_bot"].values[0]
        red_bot = player["red_playername_bot"].values[0]

        rating_a = player_elo.get(blue_bot, 1500)
        rating_b = player_elo.get(red_bot, 1500)

        region_a = region_strength2025.get(player_region.get(blue_bot, "LEC"))
        region_b = region_strength2025.get(player_region.get(red_bot, "LEC"))
        weight = (region_a / region_b) ** 0.5

        score_a = player.iloc[0]["blue_result_bot"]
        try:
            elos = updateElo(rating_a, rating_b, score_a, weight, 30)
        except:
            pass
        player_elo[blue_bot] = elos[0]
        player_elo[red_bot] = elos[1]

        #support

        blue_sup = player["blue_playername_sup"].values[0]
        red_sup = player["red_playername_sup"].values[0]

        rating_a = player_elo.get(blue_sup, 1500)
        rating_b = player_elo.get(red_sup, 1500)

        region_a = region_strength2025.get(player_region.get(blue_sup, "LEC"))
        region_b = region_strength2025.get(player_region.get(red_sup, "LEC"))
        weight = (region_a / region_b) ** 0.5

        score_a = player.iloc[0]["blue_result_sup"]
        try:
            elos = updateElo(rating_a, rating_b, score_a, weight, 30)
        except:
            pass
        player_elo[blue_sup] = elos[0]
        player_elo[red_sup] = elos[1]
    calcWinrate(players2025)
    for gameid, player in players.groupby("gameid"):
        blue_top = player["blue_playername_top"].values[0]
        blue_jng = player["blue_playername_jng"].values[0]
        blue_mid = player["blue_playername_mid"].values[0]
        blue_bot = player["blue_playername_bot"].values[0]
        blue_sup = player["blue_playername_sup"].values[0]

        red_top = player["red_playername_top"].values[0]
        red_jng = player["red_playername_jng"].values[0]
        red_mid = player["red_playername_mid"].values[0]
        red_bot = player["red_playername_bot"].values[0]
        red_sup = player["red_playername_sup"].values[0]

        players.loc[player.index, "blue_elo_top"] = player_elo.get(blue_top)
        players.loc[player.index, "blue_elo_jng"] = player_elo.get(blue_jng)
        players.loc[player.index, "blue_elo_mid"] = player_elo.get(blue_mid)
        players.loc[player.index, "blue_elo_bot"] = player_elo.get(blue_bot)
        players.loc[player.index, "blue_elo_sup"] = player_elo.get(blue_sup)

        players.loc[player.index, "blue_wr_top"] = player_blue_winrate.get(blue_top)
        players.loc[player.index, "blue_wr_jng"] = player_blue_winrate.get(blue_jng)
        players.loc[player.index, "blue_wr_mid"] = player_blue_winrate.get(blue_mid)
        players.loc[player.index, "blue_wr_bot"] = player_blue_winrate.get(blue_bot)
        players.loc[player.index, "blue_wr_sup"] = player_blue_winrate.get(blue_sup)

        players.loc[player.index, "red_elo_top"] = player_elo.get(red_top)
        players.loc[player.index, "red_elo_jng"] = player_elo.get(red_jng)
        players.loc[player.index, "red_elo_mid"] = player_elo.get(red_mid)
        players.loc[player.index, "red_elo_bot"] = player_elo.get(red_bot)
        players.loc[player.index, "red_elo_sup"] = player_elo.get(red_sup)

        players.loc[player.index, "red_wr_top"] = player_red_winrate.get(red_top)
        players.loc[player.index, "red_wr_jng"] = player_red_winrate.get(red_jng)
        players.loc[player.index, "red_wr_mid"] = player_red_winrate.get(red_mid)
        players.loc[player.index, "red_wr_bot"] = player_red_winrate.get(red_bot)
        players.loc[player.index, "red_wr_sup"] = player_red_winrate.get(red_sup)

def calcWinrate(players):
    for player in player_elo:
        current = players[players["playername"] == player]
        curr_blue = current[current["side"] == "Blue"]
        if not curr_blue.empty:
            blue_wr = curr_blue["result"].mean()
        else:
            blue_wr = 0.0
        curr_red = current[current["side"] == "Red"]
        if not curr_red.empty:
            red_wr = curr_red["result"].mean()
        else:
            red_wr = 0.0
        player_blue_winrate[player] = round(float(blue_wr), 3)
        player_red_winrate[player] = round(float(red_wr), 3)

initializePlayers(player_matches)
#%%

for gameid, player in blue_top.groupby("gameid"):
    blue_player = player["blue_top_playername"].values[0]
    blue_top.loc[player.index, "blue_top_elo"] = player_elo.get(blue_player)
    blue_top.loc[player.index, "blue_top_wr"] = player_blue_winrate.get(blue_player)
for gameid, player in blue_jng.groupby("gameid"):
    blue_player = player["blue_jng_playername"].values[0]
    blue_jng.loc[player.index, "blue_jng_elo"] = player_elo.get(blue_player)
    blue_jng.loc[player.index, "blue_jng_wr"] = player_blue_winrate.get(blue_player)
for gameid, player in blue_mid.groupby("gameid"):
    blue_player = player["blue_mid_playername"].values[0]
    blue_mid.loc[player.index, "blue_mid_elo"] = player_elo.get(blue_player)
    blue_mid.loc[player.index, "blue_mid_wr"] = player_blue_winrate.get(blue_player)
for gameid, player in blue_bot.groupby("gameid"):
    blue_player = player["blue_bot_playername"].values[0]
    blue_bot.loc[player.index, "blue_bot_elo"] = player_elo.get(blue_player)
    blue_bot.loc[player.index, "blue_bot_wr"] = player_blue_winrate.get(blue_player)
for gameid, player in blue_sup.groupby("gameid"):
    blue_player = player["blue_sup_playername"].values[0]
    blue_sup.loc[player.index, "blue_sup_elo"] = player_elo.get(blue_player)
    blue_sup.loc[player.index, "blue_sup_wr"] = player_blue_winrate.get(blue_player)

for gameid, player in red_top.groupby("gameid"):
    red_player = player["red_top_playername"].values[0]
    red_top.loc[player.index, "red_top_elo"] = player_elo.get(red_player)
    red_top.loc[player.index, "red_top_wr"] = player_red_winrate.get(red_player)
for gameid, player in red_jng.groupby("gameid"):
    red_player = player["red_jng_playername"].values[0]
    red_jng.loc[player.index, "red_jng_elo"] = player_elo.get(red_player)
    red_jng.loc[player.index, "red_jng_wr"] = player_red_winrate.get(red_player)
for gameid, player in red_mid.groupby("gameid"):
    red_player = player["red_mid_playername"].values[0]
    red_mid.loc[player.index, "red_mid_elo"] = player_elo.get(red_player)
    red_mid.loc[player.index, "red_mid_wr"] = player_red_winrate.get(red_player)
for gameid, player in red_bot.groupby("gameid"):
    red_player = player["red_bot_playername"].values[0]
    red_bot.loc[player.index, "red_bot_elo"] = player_elo.get(red_player)
    red_bot.loc[player.index, "red_bot_wr"] = player_red_winrate.get(red_player)
for gameid, player in red_sup.groupby("gameid"):
    red_player = player["red_sup_playername"].values[0]
    red_sup.loc[player.index, "red_sup_elo"] = player_elo.get(red_player)
    red_sup.loc[player.index, "red_sup_wr"] = player_red_winrate.get(red_player)


blue_matches_no_data = pd.merge(blue_top, blue_jng, on="gameid", suffixes=("",""))
blue_matches_no_data = pd.merge(blue_matches_no_data, blue_mid, on="gameid", suffixes=("",""))
blue_matches_no_data = pd.merge(blue_matches_no_data, blue_bot, on="gameid", suffixes=("",""))
blue_matches_no_data = pd.merge(blue_matches_no_data, blue_sup, on="gameid", suffixes=("",""))


red_matches_no_data = pd.merge(red_top, red_jng, on="gameid", suffixes=("",""))
red_matches_no_data = pd.merge(red_matches_no_data, red_mid, on="gameid", suffixes=("",""))
red_matches_no_data = pd.merge(red_matches_no_data, red_bot, on="gameid", suffixes=("",""))
red_matches_no_data = pd.merge(red_matches_no_data, red_sup, on="gameid", suffixes=("",""))

player_matches_no_data = pd.merge(blue_matches_no_data, red_matches_no_data, on="gameid", suffixes=("",""))
players_match_metadata = players2025.iloc[:, 0:8]
players_match_metadata = players_match_metadata.drop_duplicates(subset="gameid")
player_matches_no_data = pd.merge(players_match_metadata, player_matches_no_data, on="gameid", suffixes=("",""))
results_column = player_matches_no_data.pop("result")
player_matches_no_data.insert(0, "result", results_column)

#calculate elo differentials
for gameid, match in player_matches_no_data.groupby("gameid"):
        blue_team = match["blue_teamname"].values[0]
        red_team = match["red_teamname"].values[0]
        player_matches_no_data.loc[match.index, "blue_elo"] = team_elo.get(blue_team)
        player_matches_no_data.loc[match.index, "red_elo"] = team_elo.get(red_team)
        player_matches_no_data.loc[match.index, "elo_diff"] = team_elo.get(blue_team) - team_elo.get(red_team)

        blue_top = match["blue_top_playername"].values[0]
        blue_jng = match["blue_jng_playername"].values[0]
        blue_mid = match["blue_mid_playername"].values[0]
        blue_bot = match["blue_bot_playername"].values[0]
        blue_sup = match["blue_sup_playername"].values[0]

        red_top = match["red_top_playername"].values[0]
        red_jng = match["red_jng_playername"].values[0]
        red_mid = match["red_mid_playername"].values[0]
        red_bot = match["red_bot_playername"].values[0]
        red_sup = match["red_sup_playername"].values[0]

        player_matches_no_data.loc[match.index, "top_elo_diff"] = player_elo.get(blue_top) - player_elo.get(red_top)
        player_matches_no_data.loc[match.index, "jng_elo_diff"] = player_elo.get(blue_jng) - player_elo.get(red_jng)
        player_matches_no_data.loc[match.index, "mid_elo_diff"] = player_elo.get(blue_mid) - player_elo.get(red_mid)
        player_matches_no_data.loc[match.index, "bot_elo_diff"] = player_elo.get(blue_bot) - player_elo.get(red_bot)
        player_matches_no_data.loc[match.index, "sup_elo_diff"] = player_elo.get(blue_sup) - player_elo.get(red_sup)

#%%

sorted_elo = sorted(team_elo.items(), key=lambda x:x[1])
elo_string = str(sorted_elo)
print(textwrap.fill(elo_string, width=200))

training_data = matches.drop(columns=["red_result"])
result_column = training_data.pop("blue_result")
training_data.insert(0, "blue_result", result_column)

player_training_data = player_matches.drop(columns=["red_result_top", "red_result_mid", "red_result_jng", "red_result_bot", "red_result_sup", "blue_result_jng", "blue_result_mid", "blue_result_bot", "blue_result_sup"])
result_column = player_training_data.pop("blue_result_top")
player_training_data.insert(0, "blue_result_top", result_column)

ultra_train = pd.merge(player_training_data, training_data, on="gameid", suffixes=("",""))
ultra_train = ultra_train.drop(columns=["gameid"])

players_matches_export = player_matches_no_data.drop(columns=["gameid", "playoffs", "date", "game", "patch", "split", "league", "year", "blue_top_playername", "blue_jng_playername", "blue_mid_playername", "blue_bot_playername", "blue_sup_playername", "red_top_playername", "red_jng_playername", "red_mid_playername", "red_bot_playername", "red_sup_playername"])

predict_train = players_matches_export.drop(columns=["blue_teamname", "red_teamname"])

#print(predict_train.head(25).to_string(), len(predict_train))
players_matches_export.to_csv("players_export.csv")
predict_train.to_csv("predict_train.csv")
training_data.to_csv("team_training_data.csv")
player_training_data.to_csv("player_training_data.csv")