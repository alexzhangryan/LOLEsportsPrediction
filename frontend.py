import pandas as pd
import streamlit as st
import create_prediction_df

teams = pd.read_csv("teams.csv")["team"].tolist()



team1 = st.selectbox(
    label="team 1", 
    options=teams, 
    placeholder="Select a Team",
    index=None,
    )
team2 = st.selectbox(
    label="team 2", 
    options=teams, 
    placeholder="Select a Team",
    index=None
    )

if team1 != None and team2 != None:
        prediction = create_prediction_df.predict(team1, team2)
        st.write(prediction)