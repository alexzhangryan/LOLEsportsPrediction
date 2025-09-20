# %%
from matplotlib import pyplot as plt
import pandas as pd
import textwrap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.metrics import classification_report


df = pd.read_csv("match_data.csv")
df = df.drop(columns=["datacompleteness", "url", "participantid", "playerid", "teamid"])
df = df.fillna({"playername": "none"})
teams = df[df["playername"] == "none"]
teams = teams.drop(columns=["doublekills", "triplekills", "quadrakills", "pentakills", "firstbloodkill", "firstbloodassist", "firstbloodvictim", "elementaldrakes", "opp_elementaldrakes", "infernals", "mountains", "clouds", "oceans", "chemtechs", "hextechs", "dragons (type unknown)", "elders", "opp_elders", "firstherald", "atakhans", "opp_atakhans", "firstmidtower", "firsttothreetowers", "turretplates", "opp_turretplates", "damagemitigatedperminute", "damagetotowers", "gpr", "cspm", "damageshare", "earnedgoldshare", "total cs", "minionkills", "firstbaron", "champion", "playername", "position", "date", "year"])
teams = teams.iloc[:, :60]
teams2025 = teams[(teams["league"] == "LCK") | (teams["league"] == "LPL") | (teams["league"] == "LCK") | (teams["league"] == "LTA N") | (teams["league"] == "LTA S") | (teams["league"] == "LCP") | (teams["league"] == "LEC") | (teams["league"] == "EWC") | (teams["league"] == "FST") | (teams["league"] == "MSI") | (teams["league"] == "WLDs")]
players = df[df["playername"] != "none"]

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
training_data = matches.drop(columns=["gameid", "red_result"])
result_column = training_data.pop("blue_result")
training_data.insert(0, "blue_result", result_column)
elo = {}
region = {}
winrate = {}
#%%
def expectedScore(rating_a, rating_b):
    return (1/(1 + 10**((rating_b-rating_a)/400)))

def updateElo(rating_a, rating_b, score_a, weight, K):
    expected_a = expectedScore(rating_a, rating_b)
    expected_b = 1 - expected_a

    new_rating_a = int(rating_a + K * weight * (score_a - expected_a))
    new_rating_b = int(rating_b + K * weight * ((1 - score_a) - expected_b))

    return [new_rating_a, new_rating_b]

def initializeElo(matches):
    for gameid, match in matches.groupby("gameid"):
        blue_team = match["blue_teamname"].values[0]
        red_team = match["red_teamname"].values[0]

        rating_a = elo.get(blue_team, 1500)
        rating_b = elo.get(red_team, 1500)

        region_a = region_strength2025.get(region.get(blue_team, "LEC"))
        region_b = region_strength2025.get(region.get(red_team, "LEC"))
        weight = (region_a / region_b) ** 0.5

        score_a = match.iloc[0]["blue_result"]
        try:
            elos = updateElo(rating_a, rating_b, score_a, weight, 30)
        except:
            pass
        elo[blue_team] = elos[0]
        elo[red_team] = elos[1]
    for gameid, match in matches.groupby("gameid"):
        blue_team = match["blue_teamname"].values[0]
        red_team = match["red_teamname"].values[0]
        matches.loc[match.index, "blue_elo"] = elo.get(blue_team)
        matches.loc[match.index, "red_elo"] = elo.get(red_team)
        
        

#%%
def initializeTeams(teams):
    for team in teams.itertuples():
        elo[team[8]] = 1500
        if team[2] != "FST" or team[2] != "EWC" or team[2] != "MSI" or team[2] != "WLDs":
            region[team[12]] = team[2]
            

def initializeWinRates(teams):
    for team in teams.itertuples():
        winrate[team[8]] = 0
    for gameid, match in teams.groupby("gameid"):
        pass

initializeTeams(teams2025)


initializeElo(matches)
sorted_elo = sorted(elo.items())
elo_string = str(sorted_elo)
#print(textwrap.fill(elo_string, width=200))
#elo_string = str(elo)
#print(textwrap.fill(elo_string, width=200))
#print(matches.head(10).to_string)


#print(training_data.columns.tolist())
encoded_data = pd.get_dummies(training_data, dtype=int)
#print(encoded_data.to_string())
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

X = encoded_data.iloc[:, 1:]
y = encoded_data.iloc[:, 0]

#print(training_data.to_string())
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17, test_size=0.2)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(len(y))
print(len(y_pred))
rf.score(X_test, y_test)
print(classification_report(y_test, y_pred))
features = pd.DataFrame({
    "importance": rf.feature_importances_, 
    "feature": X.columns
})
sorted_features = features.sort_values(by="importance", ascending=False)
print(sorted_features.to_string())

#%%