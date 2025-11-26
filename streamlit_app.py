import streamlit as st
from quakefeeds import QuakeFeed
import pandas as pd
import plotly.express as px

st.title("Terremoto app")

feed = QuakeFeed("2.5","month")

token_id = "pk.eyJ1IjoibWVjb2JpIiwiYSI6IjU4YzVlOGQ2YjEzYjE3NTcxOTExZTI2OWY3Y2Y1ZGYxIn0.LUg7xQhGH2uf3zA57szCyw"

px.set_mapbox_access_token(token_id)

longitudes = [feed.location(i)[0] for i in range(len(feed))]

latitudes = [feed.location(i)[1] for i in range(len(feed))]

date = list(feed.event_times)

depths = list(feed.depths)

places = list(feed.places)

magnitudes = list(feed.magnitudes)

df = pd.DataFrame([date,longitudes,latitudes,places,magnitudes,depths]).transpose()

df.columns = ["fecha","lon","lat","loc","mag","prof"]

df["lat"] = pd.to_numeric(df["lat"])
df["lon"] = pd.to_numeric(df["lon"])
df["mag"] = pd.to_numeric(df["mag"])
df["prof"] = pd.to_numeric(df["prof"])

fig = px.scatter_mapbox(df , lat = "lat" ,lon = "lon" ,color = "mag" , size = "mag" , center = dict(lat = 18.25178 , lon = -66.254512), mapbox_style = "dark")

st.write(fig)