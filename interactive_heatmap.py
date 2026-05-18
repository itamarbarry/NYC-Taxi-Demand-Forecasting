import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import joblib
import copy
from pathlib import Path
import json
import datetime

# Import feature pipeline
from src.features import run_feature_pipeline, TARGET_COL

# ── Configuration ─────────────────────────────────────────────────────────────
GEOJSON_PATH = Path("data/interactive_heatmap/NYC_Taxi_Zones_20260512.geojson")
MODEL_DIR_TUNED = Path("models/tuned")
MODEL_DIR_ENG = Path("models/engineered")

# ── Helpers ───────────────────────────────────────────────────────────────────

def round_coords(obj, precision=5):
    """Recursively round coordinates in a GeoJSON-like dict."""
    if isinstance(obj, (int, float)):
        return round(float(obj), precision)
    if isinstance(obj, list):
        return [round_coords(x, precision) for x in obj]
    if isinstance(obj, dict):
        return {k: round_coords(v, precision) for k, v in obj.items()}
    return obj

@st.cache_resource
def load_assets():
    """Find and load the latest model and preprocessing components."""
    # 1. Find latest model
    model_files = list(MODEL_DIR_TUNED.glob("*.pkl"))
    if not model_files:
        st.error(f"No models found in {MODEL_DIR_TUNED}")
        st.stop()
    
    # Sort by modification time
    latest_model_path = max(model_files, key=lambda p: p.stat().st_mtime)
    
    # 2. Load components
    model = joblib.load(latest_model_path)
    scaler = joblib.load(MODEL_DIR_ENG / "scaler.pkl")
    mappings = joblib.load(MODEL_DIR_ENG / "mappings.joblib")
    
    # 3. Load and Optimize GeoJSON
    with open(GEOJSON_PATH, "r") as f:
        geojson_data = json.load(f)
    
    # Optimize GeoJSON by rounding coordinates to 5 decimals (~1m precision)
    geojson_data = round_coords(geojson_data)
        
    # Extract Zone ID to Name mapping
    zone_lookup = {
        str(feature["properties"].get("locationid")): feature["properties"].get("zone", "Unknown")
        for feature in geojson_data["features"]
    }

    return model, latest_model_path.name, scaler, mappings, geojson_data, zone_lookup

def get_predictions(model, scaler, mappings, input_time, input_date):
    """Generate demand predictions for all 263 taxi zones."""
    # PULocationIDs 1 to 263
    location_ids = np.arange(1, 264)
    
    # Create base datetime and round to 30-min bucket (matching training logic)
    dt = datetime.datetime.combine(input_date, input_time)
    rounded_dt = pd.Series([dt]).dt.round('30min').iloc[0]
    
    # Create base dataframe using rounded values
    df = pd.DataFrame({
        "PULocationID": location_ids,
        "hour": rounded_dt.hour,
        "day_of_week": (rounded_dt.weekday() + 1) % 7 + 1, # 1=Sun...7=Sat
        "month": rounded_dt.month,
        "pickup_datetime": rounded_dt
    })
    
    # Run feature engineering
    # Note: run_feature_pipeline returns (df, scaler, mappings) if is_training=True
    # but we want is_training=False, which returns (df, None)
    X, _ = run_feature_pipeline(df, scaler=scaler, mappings=mappings, is_training=False)
    
    # Predictions
    preds = model.predict(X)
    
    # Return as a lookup dataframe
    result = pd.DataFrame({
        "PULocationID": location_ids.astype(str), # Convert to string for Folium joining
        "predicted_demand": np.maximum(0, preds) # Ensure non-negative
    })
    return result, rounded_dt

# ── Main UI ───────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NYC Taxi Demand Heatmap",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    /* Hide Fullscreen Option and Element Toolbars */
    [data-testid="stElementToolbar"],
    [data-testid="StyledFullScreenButton"],
    [data-testid="element-fullscreen-button"],
    button[data-testid="element-fullscreen-button"],
    button[title="View fullscreen"],
    .element-fullscreen-button {
        display: none !important;
        visibility: hidden !important;
    }

    /* Hide Streamlit Header and Footer */
    header {visibility: hidden;}
    [data-testid="stHeader"] {background: rgba(0,0,0,0); height: 0;}
    #MainMenu {visibility: hidden;}
    
    .block-container {
        padding-top: 2rem !important;
    }

    /* Main Background and Typography */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Fix for unreadable text: ensure main body text is dark */
    .stMarkdown, .stText, p, li, span {
        color: #1a1a1a;
        font-size: 1.1rem !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Headers */
    h1 {
        color: #1a1a1a;
        font-family: 'Inter', sans-serif;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
        font-size: 3rem !important;
    }
    
    h2, h3 {
        color: #333333;
        font-weight: 600 !important;
        font-size: 1.5rem !important;
    }

    /* Sidebar Styling */
    .sidebar-text {
        font-size: 1.1rem;
        color: #666;
    }
    
    /* Custom Metric Cards */
    .metric-container {
        background-color: #ffffff;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #eee;
        text-align: center;
        min-height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #F7C73E;
        line-height: 1;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        background-color: #F7C73E !important;
        color: #1a1a1a !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.8rem 1rem !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        background-color: #e5b634 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(247, 199, 62, 0.3);
    }

    /* Divider Styling */
    hr {
        margin: 2rem 0 !important;
    }
    
    /* Info Box */
    .stAlert {
        border-radius: 10px;
        border: none;
        background-color: #fff9db;
    }
    </style>
    """, unsafe_allow_html=True)

# ── Title & Description ────────────────────────────────────────────────────────
col_title, col_logo = st.columns([4, 1])
with col_title:
    st.title("🚖 NYC Taxi Demand Forecast")
    st.markdown("""
        <p style='font-size: 1.1rem; color: #555; margin-bottom: 2rem;'>
        Predicting pickup demand across 263 NYC taxi zones using machine learning. Select a date and time to visualize anticipated taxi activity.
        </p>
    """, unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.header("Forecast Settings")
    
    # Date input
    today = datetime.date.today()
    selected_date = st.date_input(
        "📅 Forecast Date", 
        value=today,
        min_value=today,
        max_value=today + datetime.timedelta(days=364),
        help="Select the date for which you want to see the demand forecast."
    )
    
    # Time input
    selected_time = st.time_input("🕒 Target Time", value=datetime.time(8, 30))
    
    st.divider()
    
    # Load assets
    with st.spinner("Initializing AI Engine..."):
        model, model_name, scaler, mappings, geojson, zone_lookup = load_assets()
    
    if st.button("Generate New Forecast"):
        st.rerun()
    
    st.sidebar.write("---")
    st.sidebar.markdown(f"""
        <div style='background: #f1f3f5; padding: 15px; border-radius: 8px; border-left: 5px solid #F7C73E;'>
            <p style='margin:0; font-size: 0.85rem; color: #444;'>
            <b>Context:</b> Forecasts are aggregated into 30-minute buckets based on your selection.
            </p>
        </div>
    """, unsafe_allow_html=True)

# Generate predictions
predictions, final_rounded_dt = get_predictions(model, scaler, mappings, selected_time, selected_date)

# ── Summary Metrics ───────────────────────────────────────────────────────────
# Find Peak Zone
max_idx = predictions["predicted_demand"].idxmax()
peak_zone_id = predictions.loc[max_idx, "PULocationID"]
peak_zone_name = zone_lookup.get(peak_zone_id, "Unknown")
max_d = predictions.loc[max_idx, "predicted_demand"]

avg_d = predictions["predicted_demand"].mean()
# Zones with demand > 1
total_zones = len(predictions[predictions["predicted_demand"] > 1])

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Highest Predicted Demand</div>
            <div class="metric-value">{max_d:.1f}</div>
            <div style='color: #1a1a1a; font-size: 1.05rem; font-weight: 600; margin-bottom: 2px;'>{peak_zone_name}</div>
            <div style='color: #888; font-size: 0.9rem;'>Zone with the highest predicted pickups</div>
        </div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Average Pickup Demand</div>
            <div class="metric-value">{avg_d:.2f}</div>
            <div style='color: #1a1a1a; font-size: 1.05rem; font-weight: 600; margin-bottom: 2px;'>NYC Average</div>
            <div style='color: #888; font-size: 0.9rem;'>Average predicted pickups per zone</div>
        </div>
    """, unsafe_allow_html=True)
with m3:
    st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Active Zones</div>
            <div class="metric-value">{total_zones}</div>
            <div style='color: #1a1a1a; font-size: 1.05rem; font-weight: 600; margin-bottom: 2px;'>Active Coverage</div>
            <div style='color: #888; font-size: 0.9rem;'>Zones with predicted demand greater than 1</div>
        </div>
    """, unsafe_allow_html=True)

st.write("") # Spacer

# ── Map Rendering ─────────────────────────────────────────────────────────────

# NYC Center
NYC_COORD = [40.7128, -74.0060]

# Map PULocationID (str) to demand
demand_map = predictions.set_index("PULocationID")["predicted_demand"].to_dict()
max_demand = float(predictions["predicted_demand"].max())
max_demand = max_demand if max_demand > 0 else 1.0

# Create a deep copy of geojson to avoid modifying the cached resource
geojson_to_render = copy.deepcopy(geojson)

# Prepare GeoJSON properties for PyDeck (RGB colors)
def get_color(val, max_val):
    ratio = val / max_val
    # Taxi Yellow (#F7C73E) to Deep Red (#D32F2F)
    # Start: 247, 199, 62
    # End: 211, 47, 47
    r = int(247 + (211 - 247) * ratio)
    g = int(199 + (47 - 199) * ratio)
    b = int(62 + (47 - 62) * ratio)
    return [r, g, b, 180] # RGBA

for feature in geojson_to_render["features"]:
    loc_id_str = str(feature["properties"].get("locationid", ""))
    zone_name = feature["properties"].get("zone", "Unknown")
    demand = float(demand_map.get(loc_id_str, 0.0))
    
    feature["properties"]["demand"] = demand
    feature["properties"]["fill_color"] = get_color(demand, max_demand)
    feature["properties"]["tooltip_text"] = f"{zone_name}: {demand:.1f} pickups"

# Define PyDeck Layer
layer = pdk.Layer(
    "GeoJsonLayer",
    geojson_to_render,
    pickable=True,
    stroked=True,
    filled=True,
    extruded=False,
    wireframe=True,
    get_fill_color="properties.fill_color",
    get_line_color=[255, 255, 255],
    line_width_min_pixels=0.5,
)

# Set the viewport location
view_state = pdk.ViewState(
    latitude=NYC_COORD[0],
    longitude=NYC_COORD[1],
    zoom=10.2,
    pitch=0
)

# Render PyDeck
st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"html": "<b>{tooltip_text}</b>", "style": {"color": "white", "borderRadius": "8px"}},
    map_style="mapbox://styles/mapbox/light-v10"
))

# Footer
st.markdown("---")
st.markdown(f"""
    <div style='text-align: center; color: #888; font-size: 1rem; margin-top: 2rem;'>
        <b>NYC Taxi Demand Forecaster</b> • Predictions for {selected_date.strftime('%A, %b %d')} at {final_rounded_dt.strftime('%I:%M %p')} 
        <br>Built with Streamlit & PyDeck 
    </div>
""", unsafe_allow_html=True)
