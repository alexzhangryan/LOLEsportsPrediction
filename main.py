# %%
from re import L
import pandas as pd

df = pd.read_csv("match_data.csv")
df = df.drop(columns=["datacompleteness", "url", "participantid", "playerid", "teamid", "dragons (type unknown)"])
df = df.fillna({"playername": "none"})
teams = df[df["playername"] == "none"]
teams = teams[(teams["league"] == "LCK") | (teams["league"] == "LPL") | (teams["league"] == "LCK") | (teams["league"] == "LTA N") | (teams["league"] == "LTA S") | (teams["league"] == "LCP") | (teams["league"] == "LEC")| (teams["league"] == "Asia Master") | (teams["league"] == "EWC") | (teams["league"] == "FST") | (teams["league"] == "MSI")]
players = df[df["playername"] != "none"]

blue = teams[teams["side"] == "Blue"].copy()
print(type(teams))
red = teams[teams["side"] == "Red"].copy()
blue = blue.add_prefix("blue_")
red = red.add_prefix("red_")

blue = blue.rename(columns={"blue_gameid": "gameid"})
red = red.rename(columns={"red_gameid": "gameid"})

matches = pd.merge(blue, red, on="gameid", suffixes=("",""))

for gameid, match in matches.groupby("gameid"):
    blue_team = match["blue_teamname"]
    red_team = match["red_teamname"]
    print(blue_team)
    break


elo = {}
winrate = {}
#%%
def expectedScore(rating_a, rating_b):
    return (1/(1 + 10**((rating_b-rating_a)/400)))

def updateElo(rating_a, rating_b, score_a, K):
    expected_a = expectedScore(rating_a, rating_b)
    expected_b = 1 - expected_a

    new_rating_a = int(rating_a + K * (score_a - expected_a))
    new_rating_b = int(rating_b + K * ((1 - score_a) - expected_b))

    return [new_rating_a, new_rating_b]

def initializeElo(matches):
    for gameid, match in matches.groupby("gameid"):
        blue_team = match["blue_teamname"].values[0]
        red_team = match["red_teamname"].values[0]

        rating_a = elo.get(blue_team, 1500)
        rating_b = elo.get(red_team, 1500)

        score_a = match.iloc[0]["blue_result"]
        try:
            elos = updateElo(rating_a, rating_b, score_a, 30)
        except:
            pass
        elo[blue_team] = elos[0]
        elo[red_team] = elos[1]

#%%
def initializeTeams(teams):
    for team in teams.itertuples():
        elo[team[12]] = 1500

def initializeWinRates(teams):
    for team in teams.itertuples():
        winrate[team[12]] = 0
    for gameid, match in teams.groupby("gameid"):
        print(match)

initializeTeams(teams)
print(elo)
initializeElo(matches)
print(elo)


# %%
