"""
🌾 Smart Irrigation Assistant - COLORFUL + SMS (Fast2SMS) VERSION
- Gradient header
- Colorful AI Water Prediction card
- "📱 Send SMS Alert" button (uses Fast2SMS) — API key left as placeholder
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date, datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Irrigation Assistant", layout="wide", page_icon="🌾")

# ---------------- CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* {font-family: 'Inter', sans-serif;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.block-container {padding: 1.5rem 2rem; max-width: 1400px;}
.stMetric {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.stMetric label {color: white !important; font-weight: 600 !important;}
.stMetric [data-testid="stMetricValue"] {color: white !important; font-size: 2rem !important;}
.status-success {
    background: #10b981;
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    display: inline-block;
    font-weight: 600;
}
.status-warning {
    background: #f59e0b;
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    display: inline-block;
    font-weight: 600;
}
.status-error {
    background: #ef4444;
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    display: inline-block;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------------- CONFIG ----------------
GSHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT6L-TSWoR2thz1RBWEQ2LKv_wgjiVEdimA1sJ456_ePUaouT0hQ7iUD1mjaxJ1m0g2vcD1QZN4ybFT/pub?gid=0&single=true&output=csv"

CROP_DATA = {
    "Rice": {"early_days": 30, "mid_days": 60},
    "Wheat": {"early_days": 25, "mid_days": 50},
    "Maize": {"early_days": 20, "mid_days": 40},
    "Cotton": {"early_days": 30, "mid_days": 60},
    "Sugarcane": {"early_days": 35, "mid_days": 90},
    "Tomato": {"early_days": 25, "mid_days": 40},
}

# ---------------- DEMO DATA GENERATOR ----------------
def generate_demo_data(num_rows=100):
    now = datetime.now()
    timestamps = [now - timedelta(hours=i) for i in range(num_rows-1, -1, -1)]
    
    np.random.seed(42)
    data = {
        'Timestamp': timestamps,
        'Temperature': np.random.normal(28, 3, num_rows).clip(20, 40),
        'Humidity': np.random.normal(65, 10, num_rows).clip(30, 90),
        'Soil_Moisture': np.random.normal(1800, 300, num_rows).clip(800, 2500),
        'Rainfall': np.random.exponential(2, num_rows).clip(0, 20)
    }
    return pd.DataFrame(data)

# ---------------- CUSTOM ML MODEL ----------------
class IrrigationMLModel:
    def __init__(self):
        self.feature_weights = {
            'temperature': 0.3,
            'humidity': 0.25,
            'soil_moisture': 0.35,
            'rainfall': 0.1
        }
        self.optimal_ranges = {
            'temperature': {'optimal': 28},
            'humidity': {'optimal': 60},
            'soil_moisture': {'optimal': 1800},
            'rainfall': {'optimal': 5}
        }

    def predict(self, temperature, humidity, soil_moisture, rainfall, crop_type, days_after_sowing, area_acres):
        base_water_mm = 5.0
        crop_info = CROP_DATA.get(crop_type, CROP_DATA["Rice"])
        
        if days_after_sowing < crop_info['early_days']:
            phase_multiplier = 0.7
            phase_name = 'Early Growth'
        elif days_after_sowing < crop_info['mid_days']:
            phase_multiplier = 1.4
            phase_name = 'Mid Growth'
        else:
            phase_multiplier = 0.85
            phase_name = 'Late Growth'

        temp_penalty = ((temperature - 28)/10)**1.5 * 3 if temperature > 28 else -((28-temperature)/10) * 1.5
        humid_penalty = ((60 - humidity)/20) * 2
        soil_penalty = ((1800 - soil_moisture)/500)**1.3 * 4 if soil_moisture < 1800 else -((soil_moisture-1800)/500) * 2
        rainfall_reduction = rainfall * 0.8

        water_mm = base_water_mm * phase_multiplier
        water_mm += temp_penalty * self.feature_weights['temperature'] * 10
        water_mm += humid_penalty * self.feature_weights['humidity'] * 10
        water_mm += soil_penalty * self.feature_weights['soil_moisture'] * 10
        water_mm -= rainfall_reduction * self.feature_weights['rainfall'] * 5
        water_mm += np.random.normal(0, 0.3)
        water_mm = np.clip(water_mm, 0, 15)
        
        water_liters = water_mm * area_acres * 4046.86
        
        return {
            'water_mm': round(water_mm, 2),
            'water_liters': round(water_liters, 0),
            'phase': phase_name,
            'recommendation': 'Irrigation Required' if water_mm > 3 else 'Adequate Moisture',
            'urgency': 'High' if water_mm > 8 else ('Medium' if water_mm > 5 else 'Low')
        }

ml_model = IrrigationMLModel()

# ---------------- LOAD SENSOR DATA ----------------
@st.cache_data(ttl=300, show_spinner=False)
def load_sensor_data(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            df = pd.read_csv(url, timeout=10)
            df.columns = df.columns.str.strip()
            col_mapping = {}
            for col in df.columns:
                col_lower = col.lower()
                if 'time' in col_lower or 'date' in col_lower:
                    col_mapping[col] = 'Timestamp'
                elif 'temp' in col_lower:
                    col_mapping[col] = 'Temperature'
                elif 'humid' in col_lower:
                    col_mapping[col] = 'Humidity'
                elif 'soil' in col_lower and 'moist' in col_lower:
                    col_mapping[col] = 'Soil_Moisture'
                elif 'rain' in col_lower or 'water' in col_lower or 'precip' in col_lower:
                    col_mapping[col] = 'Rainfall'
                else:
                    col_mapping[col] = col
            df = df.rename(columns=col_mapping)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            for col in ['Temperature', 'Humidity', 'Soil_Moisture', 'Rainfall']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=['Timestamp']).sort_values('Timestamp').reset_index(drop=True)
            if len(df) > 0:
                return df, "live"
        except:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
    return generate_demo_data(), "demo"

with st.spinner("Loading sensor data..."):
    df, data_source = load_sensor_data(GSHEET_URL)

if df.empty:
    st.stop()

latest = df.iloc[-1]

# ---------------- HEADER (GRADIENT) ----------------
col_header1, col_header2 = st.columns([3, 1])
with col_header1:
    st.markdown("""
    <h1 style="
      background: linear-gradient(90deg, #ff6b6b, #facc15, #3b82f6);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      font-size: 3rem;
      font-weight: 700;
      margin-bottom:0;
    ">
    🌾 Smart Irrigation Assistant
    </h1>
    """, unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; font-size:1.1rem;'>AI-powered precision water management</p>", unsafe_allow_html=True)
with col_header2:
    if data_source == "live":
        st.success("🟢 Live Data")

st.markdown("---")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Field Configuration")
    crop_type = st.selectbox("🌾 Crop Type", list(CROP_DATA.keys()), index=0)
    sowing_date = st.date_input(
        "📅 Sowing Date",
        value=date.today() - timedelta(days=35),
        min_value=date.today() - timedelta(days=365),
        max_value=date.today()
    )
    area_acres = st.number_input("📏 Area (Acres)", min_value=0.1, max_value=1000.0, value=2.5, step=0.5)
    st.markdown("---")
    st.header("📍 Location")
    latitude = st.number_input("Latitude", value=9.9252, format="%.6f")
    longitude = st.number_input("Longitude", value=78.1198, format="%.6f")
    st.markdown("---")
    st.caption("💡 Tip: Adjust parameters to see real-time predictions")

    # ---------------- SMS SETTINGS (SIDEBAR) ----------------
    st.markdown("---")
    st.subheader("📱 SMS Alert (Fast2SMS)")
    fast2sms_key = st.text_input("Fast2SMS API Key (leave blank for placeholder)", type="password", placeholder="Paste your Fast2SMS API key here")
    sms_number = st.text_input("Recipient phone (include country code)", value="+918608508638")
    st.caption("SMS will be sent only when you click the Send SMS Alert button on the main page.")

days_after_sowing = (date.today() - sowing_date).days
prediction = ml_model.predict(
    latest['Temperature'],
    latest['Humidity'],
    latest['Soil_Moisture'],
    latest['Rainfall'],
    crop_type,
    days_after_sowing,
    area_acres
)
import os
import requests

# ---------------- SMS FUNCTION ----------------
def send_sms_via_fast2sms(message: str, number: str):
    """
    Sends SMS via Fast2SMS bulkV2 endpoint using the API key stored in environment variables.
    message: message string
    number: recipient number in format '91XXXXXXXXXX' or '+91XXXXXXXXXX'
    """
    api_key = os.getenv("FAST2SMS_API_KEY")  # 🔹 Reads the key from environment variable
    if not api_key:
        return False, "⚠️ API key not found. Please set it using: setx FAST2SMS_API_KEY 'your_key'"

    # Clean number format
    cleaned_number = number.strip().lstrip('+')

    url = "https://www.fast2sms.com/dev/bulkV2"
    payload = {
        'sender_id': "FSTSMS",
        'message': message,
        'language': "english",
        'route': "v3",
        'numbers': cleaned_number
    }
    headers = {
        'authorization': api_key,
        'Content-Type': "application/x-www-form-urlencoded"
    }

    try:
        resp = requests.post(url, data=payload, headers=headers, timeout=15)
        if resp.status_code == 200:
            try:
                j = resp.json()
                if j.get("return") is True or j.get("status") in ("success", "OK", "SUCCESS"):
                    return True, "✅ SMS sent successfully."
                return True, f"Sent (response): {j}"
            except Exception:
                return True, "✅ SMS POST succeeded (200)."
        else:
            return False, f"⚠️ HTTP {resp.status_code}: {resp.text}"
    except Exception as e:
        return False, f"❌ Exception: {e}"


# ---------------- CURRENT METRICS ----------------
st.subheader("📊 Current Sensor Readings")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="🌡️ Temperature", value=f"{latest['Temperature']:.1f} °C", delta=f"{latest['Temperature']-28:.1f}° from optimal")
with col2:
    st.metric(label="💧 Humidity", value=f"{latest['Humidity']:.1f} %", delta=f"{latest['Humidity']-60:.1f}% from optimal")
with col3:
    st.metric(label="🌱 Soil Moisture", value=f"{latest['Soil_Moisture']:.0f} mV", delta=f"{latest['Soil_Moisture']-1800:.0f} from optimal")
with col4:
    st.metric(label="🌧️ Rainfall", value=f"{latest['Rainfall']:.1f} mm", delta="Last hour")

# ---------------- AI WATER PREDICTION (COLORFUL CARD) ----------------
st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #34d399, #3b82f6, #facc15);
    color: white;
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
">
    <h2 style='font-size:2.5rem; font-weight:700; margin-bottom:0.5rem;'>💧 AI Water Requirement Prediction</h2>
    <p style='font-size:3rem; font-weight:800; margin:1rem 0; background: linear-gradient(to right, #ff6b6b, #facc15, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>{prediction['water_liters']:,.0f} Liters</p>
    <p style='font-size:1.5rem; margin-bottom:1rem;'>({prediction['water_mm']} mm per acre)</p>
    <p style='font-size:1.2rem; margin-top:1rem;'><strong>Growth Phase:</strong> {prediction['phase']}</p>
    <p style='font-size:1.2rem;'><strong>Status:</strong> {prediction['recommendation']}</p>
    <p style='font-size:1.2rem;'><strong>Urgency:</strong> {prediction['urgency']}</p>
</div>
""", unsafe_allow_html=True)


# ---------------- SENSOR TRENDS ----------------
st.subheader("📈 Sensor Trends (Last 48 Hours)")
recent = df.tail(48) if len(df) >= 48 else df

fig_trend = go.Figure()

fig_trend.add_trace(go.Scatter(
    x=recent['Timestamp'],
    y=recent['Soil_Moisture'],
    mode='lines+markers',
    name='Soil Moisture',
    line=dict(color='#8b5cf6', width=3),
    marker=dict(size=6)
))

fig_trend.add_trace(go.Scatter(
    x=recent['Timestamp'],
    y=recent['Temperature']*50,
    mode='lines+markers',
    name='Temperature (×50)',
    line=dict(color='#ef4444', width=2),
    yaxis='y2'
))

fig_trend.add_trace(go.Scatter(
    x=recent['Timestamp'],
    y=recent['Humidity']*20,
    mode='lines+markers',
    name='Humidity (×20)',
    line=dict(color='#3b82f6', width=2),
    yaxis='y2'
))

fig_trend.add_trace(go.Bar(
    x=recent['Timestamp'],
    y=recent['Rainfall']*100,
    name='Rainfall (×100)',
    marker_color='#10b981',
    opacity=0.4,
    yaxis='y2'
))

fig_trend.update_layout(
    height=500,
    xaxis_title="Time",
    yaxis_title="Soil Moisture (mV)",
    yaxis2=dict(title="Scaled Values", overlaying='y', side='right'),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white",
    hovermode='x unified'
)

st.plotly_chart(fig_trend, use_container_width=True)

# ---------------- TWO COLUMN LAYOUT ----------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("💧 Soil Moisture Distribution")
    df['Moisture_Status'] = df['Soil_Moisture'].apply(
        lambda m: 'Dry (<1000)' if m < 1000 else ('Optimal (1000-2000)' if m <= 2000 else 'Wet (>2000)')
    )
    moisture_counts = df['Moisture_Status'].value_counts().reset_index()
    moisture_counts.columns = ['status', 'count']
    
    fig_pie = px.pie(
        moisture_counts,
        names='status',
        values='count',
        color='status',
        color_discrete_map={
            'Dry (<1000)': '#ff6b6b',
            'Optimal (1000-2000)': '#51cf66',
            'Wet (>2000)': '#339af0'
        },
        hole=0.4
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(height=400)
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("📊 Data Summary")
    summary_data = {
        'Metric': ['Temperature', 'Humidity', 'Soil Moisture', 'Rainfall'],
        'Average': [
            f"{df['Temperature'].mean():.1f} °C",
            f"{df['Humidity'].mean():.1f} %",
            f"{df['Soil_Moisture'].mean():.0f} mV",
            f"{df['Rainfall'].mean():.2f} mm"
        ],
        'Min': [
            f"{df['Temperature'].min():.1f}",
            f"{df['Humidity'].min():.1f}",
            f"{df['Soil_Moisture'].min():.0f}",
            f"{df['Rainfall'].min():.2f}"
        ],
        'Max': [
            f"{df['Temperature'].max():.1f}",
            f"{df['Humidity'].max():.1f}",
            f"{df['Soil_Moisture'].max():.0f}",
            f"{df['Rainfall'].max():.2f}"
        ]
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

# ---------------- WEATHER FORECAST ----------------
st.subheader("🌤️ 3-Day Weather Forecast")

@st.cache_data(ttl=3600)
def get_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto&forecast_days=3"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        df_weather = pd.DataFrame(data['daily'])
        df_weather['date'] = pd.to_datetime(df_weather['time'])
        return df_weather[['date', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum']]
    except:
        return pd.DataFrame()

forecast = get_weather(latitude, longitude)

if not forecast.empty:
    fig_fc = go.Figure()
    
    fig_fc.add_trace(go.Bar(
        x=forecast['date'],
        y=forecast['precipitation_sum'],
        name='Precipitation (mm)',
        marker_color='#3b82f6',
        yaxis='y2'
    ))
    
    fig_fc.add_trace(go.Scatter(
        x=forecast['date'],
        y=forecast['temperature_2m_max'],
        name='Max Temp (°C)',
        line=dict(color='#ef4444', width=3),
        mode='lines+markers',
        marker=dict(size=10)
    ))
    
    fig_fc.add_trace(go.Scatter(
        x=forecast['date'],
        y=forecast['temperature_2m_min'],
        name='Min Temp (°C)',
        line=dict(color='#f97316', width=3),
        mode='lines+markers',
        marker=dict(size=10)
    ))
    
    fig_fc.update_layout(
        yaxis_title="Temperature (°C)",
        yaxis2=dict(title="Precipitation (mm)", overlaying='y', side='right'),
        xaxis_title="Date",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )
    
    st.plotly_chart(fig_fc, use_container_width=True)
else:
    st.warning("⚠️ Weather forecast temporarily unavailable")

# ---------------- FARM LOCATION ----------------
st.subheader("📍 Farm Location")
m = folium.Map(location=[latitude, longitude], zoom_start=14, tiles='OpenStreetMap')
folium.Marker(
    [latitude, longitude],
    popup=f"<b>{crop_type} Farm</b><br>{area_acres} acres",
    tooltip="Your Farm",
    icon=folium.Icon(color='green', icon='leaf', prefix='fa')
).add_to(m)

folium.Circle(
    [latitude, longitude],
    radius=500,
    color='green',
    fill=True,
    opacity=0.3
).add_to(m)

st_folium(m, width=None, height=450)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#64748b; font-size:0.9rem;'>"
    "🌾 Smart Irrigation Assistant v2.1 | Built with ❤️ using Streamlit & AI"
    "</p>",
    unsafe_allow_html=True
)
