import streamlit as st
import pandas as pd

st.title("Titanic Survival Prediction")
st.write("By Harshal - Data Analyst")

# csv file cha nav titanic.csv asel tar hech thev
df = pd.read_csv("titanic.csv")
st.write("Dataset Preview:")
st.dataframe(df.head())
