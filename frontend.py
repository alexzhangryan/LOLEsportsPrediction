import time
import pandas as pd
import streamlit as st
import create_prediction_df

st.set_page_config(layout="wide")

if "playins_clicked" not in st.session_state:
    st.session_state.playins_clicked = False
if "swiss_clicked" not in st.session_state:
    st.session_state.swiss_clicked = False
if "playins_teams" not in st.session_state:
    st.session_state.playins_teams = [["T1", "tbd"], ["Invictus Gaming", "tbd"]]
if "swiss_10" not in st.session_state:
    st.session_state.swiss_10 = [["TBD", "tbd"] for _ in range(8)]
if "swiss_01" not in st.session_state:
    st.session_state.swiss_01 = [["TBD", "tbd"] for _ in range(8)]
if "swiss_20" not in st.session_state:
    st.session_state.swiss_20 = [["TBD", "tbd"] for _ in range(4)]
if "swiss_11" not in st.session_state:
    st.session_state.swiss_11 = [["TBD", "tbd"] for _ in range(8)]
if "swiss_02" not in st.session_state:
    st.session_state.swiss_02 = [["TBD", "tbd"] for _ in range(4)]
if "swiss_21" not in st.session_state:
    st.session_state.swiss_21 = [["TBD", "tbd"] for _ in range(6)]
if "swiss_12" not in st.session_state:
    st.session_state.swiss_12 = [["TBD", "tbd"] for _ in range(6)]
if "swiss_22" not in st.session_state:
    st.session_state.swiss_22 = [["TBD", "tbd"] for _ in range(6)]
if "swiss_advances" not in st.session_state:
    st.session_state.swiss_advances = [["TBD", "tbd"] for _ in range(8)]
if "swiss_eliminated" not in st.session_state:
    st.session_state.swiss_eliminated = [["TBD", "tbd"] for _ in range(8)]
if "semifinals" not in st.session_state:
    st.session_state.semifinals = [["TBD", "tbd"] for _ in range(4)]
if "finals" not in st.session_state:
    st.session_state.finals = [["TBD", "tbd"] for _ in range(2)]
if "champion" not in st.session_state:
    st.session_state.champion = "TBD"
if "swiss_round_1" not in st.session_state:
    st.session_state.swiss_round_1 = [
        ["Gen.G", "tbd"],
        ["TBD", "tbd"],
        ["Hanwha Life Esports", "tbd"],
        ["Vivo Keyd Stars", "tbd"],
        ["Anyone's Legend", "tbd"],
        ["Team Secret Whales", "tbd"],
        ["Bilibili Gaming", "tbd"],
        ["100 Thieves", "tbd"],
        ["FlyQuest", "tbd"],
        ["Fnatic", "tbd"],
        ["Top Esports", "tbd"],
        ["Movistar KOI", "tbd"],
        ["KT Rolster", "tbd"],
        ["PSG Talon", "tbd"],
        ["G2 Esports", "tbd"],
        ["CTBC Flying Oyster", "tbd"],
    ]
if "swiss_index" not in st.session_state:
    st.session_state.swiss_index = 1
if "knockout_index" not in st.session_state:
    st.session_state.knockout_index = 1


def print_teams(teams):
    for i in range(0, len(teams), 2):
        with st.container(border=True):
            if teams[i][1] == "win":
                st.markdown(f""":green[{teams[i][0]}]""")
            elif teams[i][1] == "loss":
                st.markdown(f""":red[{teams[i][0]}]""")
            else:
                st.write(teams[i][0])
            if teams[i + 1][1] == "win":
                st.markdown(f""":green[{teams[i + 1][0]}]""")
            elif teams[i + 1][1] == "loss":
                st.markdown(f""":red[{teams[i + 1][0]}]""")
            else:
                st.write(teams[i + 1][0])


def find_team_index(teams, name):
    return next((i for i, t in enumerate(teams) if t[0] == name), None)


def click_playin():
    st.session_state.playins_clicked = True
    prediction = create_prediction_df.predict("T1", "Invictus Gaming")
    st.session_state.playins_teams[
        find_team_index(st.session_state.playins_teams, prediction[0])
    ][1] = "win"
    st.session_state.playins_teams[
        find_team_index(st.session_state.playins_teams, prediction[1])
    ][1] = "loss"
    st.session_state.swiss_round_1[1] = [prediction[0], "tbd"]


def click_swiss():
    if st.session_state.swiss_index == 1:
        for i in range(0, len(st.session_state.swiss_round_1), 2):
            predict = create_prediction_df.predict(
                st.session_state.swiss_round_1[i][0],
                st.session_state.swiss_round_1[i + 1][0],
            )
            st.session_state.swiss_10[int(i / 2)][0] = predict[0]
            st.session_state.swiss_01[int(i / 2)][0] = predict[1]
            st.session_state.swiss_round_1[
                find_team_index(st.session_state.swiss_round_1, predict[0])
            ][1] = "win"
            st.session_state.swiss_round_1[
                find_team_index(st.session_state.swiss_round_1, predict[1])
            ][1] = "loss"
    if st.session_state.swiss_index == 2:
        for i in range(0, len(st.session_state.swiss_10), 2):
            predict = create_prediction_df.predict(
                st.session_state.swiss_10[i][0], st.session_state.swiss_10[i + 1][0]
            )
            st.session_state.swiss_20[int(i / 2)][0] = predict[0]
            st.session_state.swiss_11[int(i / 2)][0] = predict[1]
            st.session_state.swiss_10[
                find_team_index(st.session_state.swiss_10, predict[0])
            ][1] = "win"
            st.session_state.swiss_10[
                find_team_index(st.session_state.swiss_10, predict[1])
            ][1] = "loss"
        for i in range(0, len(st.session_state.swiss_01), 2):
            predict = create_prediction_df.predict(
                st.session_state.swiss_01[i][0], st.session_state.swiss_01[i + 1][0]
            )
            st.session_state.swiss_11[4 + int(i / 2)][0] = predict[0]
            st.session_state.swiss_02[int(i / 2)][0] = predict[1]
            st.session_state.swiss_01[
                find_team_index(st.session_state.swiss_01, predict[0])
            ][1] = "win"
            st.session_state.swiss_01[
                find_team_index(st.session_state.swiss_01, predict[1])
            ][1] = "loss"
    if st.session_state.swiss_index == 3:
        for i in range(0, len(st.session_state.swiss_20), 2):
            predict = create_prediction_df.predict(
                st.session_state.swiss_20[i][0], st.session_state.swiss_20[i + 1][0]
            )
            st.session_state.swiss_advances[int(i / 2)][0] = predict[0]
            st.session_state.swiss_21[int(i / 2)][0] = predict[1]
            st.session_state.swiss_20[
                find_team_index(st.session_state.swiss_20, predict[0])
            ][1] = "win"
            st.session_state.swiss_20[
                find_team_index(st.session_state.swiss_20, predict[1])
            ][1] = "loss"
        for i in range(0, len(st.session_state.swiss_11), 2):
            predict = create_prediction_df.predict(
                st.session_state.swiss_11[i][0], st.session_state.swiss_11[i + 1][0]
            )
            st.session_state.swiss_21[2 + int(i / 2)][0] = predict[0]
            st.session_state.swiss_12[int(i / 2)][0] = predict[1]
            st.session_state.swiss_11[
                find_team_index(st.session_state.swiss_11, predict[0])
            ][1] = "win"
            st.session_state.swiss_11[
                find_team_index(st.session_state.swiss_11, predict[1])
            ][1] = "loss"
        for i in range(0, len(st.session_state.swiss_02), 2):
            predict = create_prediction_df.predict(
                st.session_state.swiss_02[i][0], st.session_state.swiss_02[i + 1][0]
            )
            st.session_state.swiss_12[4 + int(i / 2)][0] = predict[0]
            st.session_state.swiss_eliminated[int(i / 2)][0] = predict[1]
            st.session_state.swiss_02[
                find_team_index(st.session_state.swiss_02, predict[0])
            ][1] = "win"
            st.session_state.swiss_02[
                find_team_index(st.session_state.swiss_02, predict[1])
            ][1] = "loss"
    if st.session_state.swiss_index == 4:
        for i in range(0, len(st.session_state.swiss_21), 2):
            predict = create_prediction_df.predict(
                st.session_state.swiss_21[i][0], st.session_state.swiss_21[i + 1][0]
            )
            st.session_state.swiss_advances[2 + int(i / 2)][0] = predict[0]
            st.session_state.swiss_22[int(i / 2)][0] = predict[1]
            st.session_state.swiss_21[
                find_team_index(st.session_state.swiss_21, predict[0])
            ][1] = "win"
            st.session_state.swiss_21[
                find_team_index(st.session_state.swiss_21, predict[1])
            ][1] = "loss"
        for i in range(0, len(st.session_state.swiss_12), 2):
            predict = create_prediction_df.predict(
                st.session_state.swiss_12[i][0], st.session_state.swiss_12[i + 1][0]
            )
            st.session_state.swiss_22[3 + int(i / 2)][0] = predict[0]
            st.session_state.swiss_eliminated[2 + int(i / 2)][0] = predict[1]
            st.session_state.swiss_12[
                find_team_index(st.session_state.swiss_12, predict[0])
            ][1] = "win"
            st.session_state.swiss_12[
                find_team_index(st.session_state.swiss_12, predict[1])
            ][1] = "loss"
    if st.session_state.swiss_index == 5:
        for i in range(0, len(st.session_state.swiss_22), 2):
            predict = create_prediction_df.predict(
                st.session_state.swiss_22[i][0], st.session_state.swiss_22[i + 1][0]
            )
            st.session_state.swiss_advances[5 + int(i / 2)][0] = predict[0]
            st.session_state.swiss_eliminated[5 + int(i / 2)][0] = predict[1]
            st.session_state.swiss_22[
                find_team_index(st.session_state.swiss_22, predict[0])
            ][1] = "win"
            st.session_state.swiss_22[
                find_team_index(st.session_state.swiss_22, predict[1])
            ][1] = "loss"
        st.session_state.swiss_clicked = True
    st.session_state.swiss_index += 1


def click_knockout():
    if st.session_state.knockout_index == 1:
        for i in range(0, len(st.session_state.swiss_advances), 2):
            predict = create_prediction_df.predict(
                st.session_state.swiss_advances[i][0],
                st.session_state.swiss_advances[i + 1][0],
            )
            st.session_state.semifinals[int(i / 2)][0] = predict[0]
            st.session_state.swiss_advances[
                find_team_index(st.session_state.swiss_advances, predict[0])
            ][1] = "win"
            st.session_state.swiss_advances[
                find_team_index(st.session_state.swiss_advances, predict[1])
            ][1] = "loss"
    if st.session_state.knockout_index == 2:
        for i in range(0, len(st.session_state.semifinals), 2):
            predict = create_prediction_df.predict(
                st.session_state.semifinals[i][0],
                st.session_state.semifinals[i + 1][0],
            )
            st.session_state.finals[int(i / 2)][0] = predict[0]
            st.session_state.semifinals[
                find_team_index(st.session_state.semifinals, predict[0])
            ][1] = "win"
            st.session_state.semifinals[
                find_team_index(st.session_state.semifinals, predict[1])
            ][1] = "loss"
    if st.session_state.knockout_index == 3:
        for i in range(0, len(st.session_state.finals), 2):
            predict = create_prediction_df.predict(
                st.session_state.finals[i][0],
                st.session_state.finals[i + 1][0],
            )
            st.session_state.champion = predict[0]
            st.session_state.finals[
                find_team_index(st.session_state.finals, predict[0])
            ][1] = "win"
            st.session_state.finals[
                find_team_index(st.session_state.finals, predict[1])
            ][1] = "loss"
    st.session_state.knockout_index += 1


teams = pd.read_csv("teams.csv")["team"].tolist()

team1 = st.selectbox(
    label="team 1",
    options=teams,
    placeholder="Select a Team",
    index=None,
)
team2 = st.selectbox(
    label="team 2", options=teams, placeholder="Select a Team", index=None
)

if team1 is not None and team2 is not None:
    prediction = create_prediction_df.predict(team1, team2)
    st.header(f"{prediction[0]} will win with {prediction[2]}% confidence")

st.button("Predict Playins", on_click=click_playin)

with st.container(horizontal=True):
    with st.container(border=True):
        st.title("Play-ins")
        with st.container(border=True):
            for team in st.session_state.playins_teams:
                if team[1] == "win":
                    st.markdown(f""":green[{team[0]}]""")
                elif team[1] == "loss":
                    st.markdown(f""":red[{team[0]}]""")
                else:
                    st.write(team[0])

        if st.session_state.playins_clicked:
            playin_prediction = create_prediction_df.predict("T1", "Invictus Gaming")

st.button(
    f"Predict Swiss Stage Round {st.session_state.swiss_index}",
    on_click=click_swiss,
    disabled=not st.session_state.playins_clicked,
)

with st.container(border=True):
    st.title("Swiss Stage")
    with st.container(horizontal=True):
        with st.container(border=True):
            st.header("0-0")
            print_teams(st.session_state.swiss_round_1)
        with st.container(border=True):
            st.header("1-0")
            print_teams(st.session_state.swiss_10)
            st.header("0-1")
            print_teams(st.session_state.swiss_01)
        with st.container(border=True):
            st.header("2-0")
            print_teams(st.session_state.swiss_20)
            st.header("1-1")
            print_teams(st.session_state.swiss_11)
            st.header("0-2")
            print_teams(st.session_state.swiss_02)
        with st.container(border=True):
            st.header("2-1")
            print_teams(st.session_state.swiss_21)
            st.header("1-2")
            print_teams(st.session_state.swiss_12)
        with st.container(border=True):
            st.header("2-2")
            print_teams(st.session_state.swiss_22)
        with st.container(border=True):
            st.header("Advances")
            for team in st.session_state.swiss_advances:
                st.write(team[0])
            st.header("Eliminated")
            for team in st.session_state.swiss_eliminated:
                st.write(team[0])

st.button(
    f"Predict Knockout Stage Round {st.session_state.knockout_index}",
    on_click=click_knockout,
    disabled=not st.session_state.swiss_clicked,
)

with st.container(border=True):
    st.title("Knockout Stage")
    with st.container(horizontal=True):
        with st.container(border=True):
            st.header("Quarterfinals")
            print_teams(st.session_state.swiss_advances)
        with st.container(border=True):
            st.header("Semifinals")
            print_teams(st.session_state.semifinals)
        with st.container(border=True):
            st.header("Finals")
            print_teams(st.session_state.finals)
if not st.session_state.champion == "TBD":
    st.title(f"Your 2025 World Champions are {st.session_state.champion}")
    st.balloons()
