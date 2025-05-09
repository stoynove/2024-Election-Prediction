import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import numpy as np

st.set_page_config(page_title="County Prediction Map", layout="wide")
st.title("🗳️ County Prediction Map")

# Load your data
df = pd.read_csv("PROJECT.csv")
df['FIPS_Code'] = df['FIPS_Code'].apply(lambda x: str(x).zfill(5))

# Train Random Forest and get top 10 features
features = [col for col in df.columns if col not in ['FIPS_Code', 'Republican']]
X = df[features]
y = df['Republican']
model = RandomForestClassifier(
    n_estimators=2000,
    oob_score=True,
    n_jobs=-1,
    random_state=42,
    max_features=0.2,
    min_samples_leaf=1
)
model.fit(X, y)
importances = pd.Series(model.feature_importances_, index=X.columns)
top10 = importances.sort_values(ascending=False).head(10).index.tolist()

# Download US counties geojson and get all FIPS codes
@st.cache_data
def get_county_geojson_and_fips():
    import json, urllib.request
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    with urllib.request.urlopen(url) as response:
        counties_geo = json.load(response)
    all_fips = [feature['id'] for feature in counties_geo['features']]
    return counties_geo, all_fips

counties_geo, all_fips = get_county_geojson_and_fips()

# Create base dataframe with all FIPS codes
all_fips_df = pd.DataFrame({'FIPS_Code': all_fips})

# Merge with your data, keeping all FIPS codes
merged = all_fips_df.merge(df[['FIPS_Code'] + features + ['Republican']], on='FIPS_Code', how='left')

# Fill missing values
for col in features:
    merged[col] = merged[col].fillna(df[col].mean())
merged['Republican'] = merged['Republican'].fillna(0)

# Store initial predictions
if 'initial_predictions' not in st.session_state:
    initial_X = merged[features].copy()
    st.session_state.initial_predictions = model.predict(initial_X)
    st.session_state.initial_values = merged[features].copy()

# Sidebar controls
st.sidebar.header("Adjust Top 10 Features (%)")

# Add reset button
if st.sidebar.button("Reset All Parameters"):
    for feature in top10:
        st.session_state[f"slider_{feature}"] = 0.0
    st.rerun()

st.sidebar.markdown("Adjust the percentage change for each feature (-50% to +50%)")

# Sliders for top 10 features
slider_values = {}
for feature in top10:
    slider_values[feature] = st.sidebar.slider(
        f"{feature} % Change",
        min_value=-50.0,
        max_value=50.0,
        value=0.0,
        format="%.1f%%",
        key=f"slider_{feature}"
    )

# Apply changes and predict
X_mod = st.session_state.initial_values.copy()
for feature in top10:
    change = slider_values[feature] / 100.0
    X_mod[feature] = st.session_state.initial_values[feature] * (1 + change)

# Make predictions
predictions = model.predict(X_mod)

# Identify counties that changed
changed_counties = predictions != st.session_state.initial_predictions
changes = np.sum(changed_counties)

# Create color array
colors = np.where(changed_counties, 2,  # 2 represents yellow for changed counties
                 predictions)  # 0 for blue (Democrat), 1 for red (Republican)

st.sidebar.markdown("---")
st.sidebar.markdown(f"### Changes in Predictions")
st.sidebar.markdown(f"Number of counties that changed: {changes}")

# Create the map with three colors
fig = go.Figure(go.Choropleth(
    geojson=counties_geo,
    locations=merged['FIPS_Code'],
    z=colors,
    colorscale=[[0, 'blue'], [0.5, 'red'], [1, 'yellow']],  # Blue for 0, Red for 1, Yellow for 2
    zmin=0,
    zmax=2,
    marker_line_width=0.5,
    showscale=False  # Remove the colorbar
))

fig.update_layout(
    title="Predicted County-Level Election Results (Interactive)",
    geo=dict(scope='usa', showlakes=True, lakecolor='rgb(255,255,255)'),
    width=1400, height=800
)

st.plotly_chart(fig)

# Show some statistics
total_rep = np.sum(predictions == 1)
total_dem = np.sum(predictions == 0)
st.markdown(f"### Current Predictions")
st.markdown(f"Republican counties: {total_rep}")
st.markdown(f"Democratic counties: {total_dem}")
st.markdown(f"Counties that changed prediction: {changes} (shown in yellow)")