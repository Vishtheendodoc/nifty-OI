import requests
import pandas as pd
import time
import streamlit as st
import plotly.express as px
import os
from datetime import datetime
import pytz

# Enhancement imports
from collections import deque

# 🔹 Set IST Timezone
IST = pytz.timezone("Asia/Kolkata")

# Streamlit Page Configuration
st.set_page_config(page_title="Nifty Options IV Spike Dashboard", layout="wide")

# 🔹 Dhan API Credentials (Replace with your own)
# ====== Dhan API Config ======
CLIENT_ID = '1100244268'
ACCESS_TOKEN= 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzQ4MDg2NzE5LCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwMDI0NDI2OCJ9.oIiH0BsTyn3bmGB5wVCVDxfQ93AVq1dDT-5nCPrVEfyGoknwajxL1Pogn5TNA3tDguz-97OhOaPB_BhECDcSYA'  # Replace with your Access Token

HEADERS = {
    'client-id': CLIENT_ID,
    'access-token': ACCESS_TOKEN,
    'Content-Type': 'application/json'
}

# 🔹 Telegram Bot Credentials (Replace with your own)
TELEGRAM_BOT_TOKEN = "7967747029:AAFyMl5zF1XvRqrhY5CIoR_1_EJwiEyrAqw"
TELEGRAM_CHAT_ID = "-470480347"

# 🔹 API Endpoints
OPTION_CHAIN_URL = "https://api.dhan.co/v2/optionchain"
EXPIRY_LIST_URL = "https://api.dhan.co/v2/optionchain/expirylist"

# 🔹 Nifty Index Code
NIFTY_SCRIP_ID = 13
NIFTY_SEGMENT = "IDX_I"

# CSV File Path
CSV_FILE = "nifty_option_chain.csv"



# Store rolling IV/OI history
if "rolling_data" not in st.session_state:
    st.session_state.rolling_data = {}

# Store previous data for comparison
if "previous_data" not in st.session_state:
    st.session_state.previous_data = {}
previous_data = st.session_state.previous_data  # Use session state storage


# Store last cycle alerts to prevent duplicates
sent_alerts = {}

# Streamlit session state for alerts
if "alerts" not in st.session_state:
    st.session_state.alerts = []

# Function to send Telegram Alerts
def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, data=data)

# Function to fetch expiry dates
def get_expiry_dates():
    url = "https://api.dhan.co/v2/optionchain/expirylist"
    payload = {"UnderlyingScrip": 13, "UnderlyingSeg": "IDX_I"}
    response = requests.post(url, json=payload, headers=HEADERS)
    if response.status_code != 200:
        st.error(f"Failed to fetch expiry list: {response.text}")
        st.stop()
    return response.json()['data']

def fetch_option_chain(expiry):
    url = "https://api.dhan.co/v2/optionchain"
    payload = {"UnderlyingScrip": 13, "UnderlyingSeg": "IDX_I", "Expiry": expiry}
    response = requests.post(url, json=payload, headers=HEADERS)
    time.sleep(3)
    if response.status_code != 200:
        st.error(f"Failed to fetch option chain: {response.text}")
        st.stop()
    return response.json()

# Function to analyze IV, OI, and Greeks
# Define Alert Thresholds
IV_SPIKE_THRESHOLD = 5  # IV increase threshold (%)
IV_CRASH_THRESHOLD = 5   # IV drop threshold (%)
OI_SPIKE_THRESHOLD = 10  # OI increase threshold (%)
GAMMA_THRESHOLD = 0.02   # Gamma exposure threshold
THETA_DECAY_THRESHOLD = -20  # Theta erosion threshold
PRICE_STABILITY_THRESHOLD = 0.5  # Price stability for IV-based alerts

def analyze_data(option_chain):
    # Ensure previous data persists across Streamlit reruns
    if "previous_data" not in st.session_state:
        st.session_state.previous_data = {}  

    previous_data = st.session_state.previous_data  # Use session state storage

    if "data" not in option_chain or "oc" not in option_chain["data"]:
        st.error("Invalid option chain data received!")
        return pd.DataFrame()

    option_chain_data = option_chain["data"]["oc"]
    data_list = []
    underlying_price = option_chain["data"]["last_price"]  # Fetch underlying price

    # Determine ATM Strike
    atm_strike = min(option_chain_data.keys(), key=lambda x: abs(float(x) - underlying_price))
    atm_strike = float(atm_strike)

    # Define range for ATM ± 4 strikes
    min_strike = atm_strike - 5 * 50  # Assuming 50-point strike intervals
    max_strike = atm_strike + 5 * 50

    for strike, contracts in option_chain_data.items():
        strike_price = float(strike)  # Convert key to float

        # Filter only ATM ± 4 strikes
        if strike_price < min_strike or strike_price > max_strike:
            continue

        ce_data = contracts.get("ce", {})
        pe_data = contracts.get("pe", {})

        ce_iv = ce_data.get("implied_volatility", 0)
        ce_oi = ce_data.get("oi", 0)
        ce_ltp = ce_data.get("last_price", 0)
        ce_delta = ce_data.get("greeks", {}).get("delta", 0)
        ce_gamma = ce_data.get("greeks", {}).get("gamma", 0)
        ce_theta = ce_data.get("greeks", {}).get("theta", 0)

        pe_iv = pe_data.get("implied_volatility", 0)
        pe_oi = pe_data.get("oi", 0)
        pe_ltp = pe_data.get("last_price", 0)
        pe_delta = pe_data.get("greeks", {}).get("delta", 0)
        pe_gamma = pe_data.get("greeks", {}).get("gamma", 0)
        pe_theta = pe_data.get("greeks", {}).get("theta", 0)

        # Add data to list
        data_list.append({
            "StrikePrice": strike_price,
            "Type": "CE",
            "IV": ce_iv,
            "OI": ce_oi,
            "LTP": ce_ltp,
            "Delta": ce_delta,
            "Gamma": ce_gamma,
            "Theta": ce_theta
        })

        data_list.append({
            "StrikePrice": strike_price,
            "Type": "PE",
            "IV": pe_iv,
            "OI": pe_oi,
            "LTP": pe_ltp,
            "Delta": pe_delta,
            "Gamma": pe_gamma,
            "Theta": pe_theta
        })

    df = pd.DataFrame(data_list)
    df["Timestamp"] = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    df = df.sort_values(by=["StrikePrice", "Type"])

    alerts = []
    prev_underlying_price = previous_data.get("underlying_price", underlying_price)

    # Update rolling data
    rolling_data = st.session_state.rolling_data
    for _, row in df.iterrows():
        key = f"{row['StrikePrice']}_{row['Type']}"
        if key not in rolling_data:
            rolling_data[key] = deque(maxlen=5)
        rolling_data[key].append({"IV": row["IV"], "OI": row["OI"], "LTP": row["LTP"]})

    # Calculate Net OI Imbalance
    net_oi_df = df.pivot(index='StrikePrice', columns='Type', values='OI').fillna(0)
    net_oi_df['NetOI'] = net_oi_df['CE'] - net_oi_df['PE']

    # Add OI Change % to DataFrame
    df["OI_Change"] = df.apply(lambda row: (
        ((row["OI"] - previous_data.get(f'{row["StrikePrice"]}_{row["Type"]}', {}).get("OI", row["OI"])) / row["OI"]) * 100
    ) if row["OI"] else 0, axis=1)

    # Compare with previous data and apply indicators
    for _, row in df.iterrows():
        strike_price = row["StrikePrice"]
        opt_type = row["Type"]
        iv = row["IV"]
        oi = row["OI"]
        ltp = row["LTP"]
        delta = row["Delta"]
        gamma = row["Gamma"]
        theta = row["Theta"]

        key = f"{strike_price}_{opt_type}"

        # Fetch previous data
        prev = previous_data.get(key, {})
        prev_iv = prev.get("IV", iv)
        prev_oi = prev.get("OI", oi)

        # IV and OI Change
        iv_change = ((iv - prev_iv) / prev_iv) * 100 if prev_iv else 0
        oi_change = ((oi - prev_oi) / prev_oi) * 100 if prev_oi else 0
        price_change = abs((underlying_price - prev_underlying_price) / prev_underlying_price * 100) if prev_underlying_price else 0

        # STRONG BREAKOUT ALERT
        if opt_type == "CE" and iv_change > IV_SPIKE_THRESHOLD and oi_change > OI_SPIKE_THRESHOLD and delta > 0.75:
            alerts.append(f"🔥 STRONG BREAKOUT (CALLS): IV + OI + Delta Confirmed!\nStrike: {strike_price} | IV: {iv_change:.2f}% | OI: {oi_change:.2f}% | Delta: {delta:.2f}")

        if opt_type == "PE" and iv_change > IV_SPIKE_THRESHOLD and oi_change > OI_SPIKE_THRESHOLD and delta < -0.75:
            alerts.append(f"🔥 STRONG BREAKOUT (PUTS): IV + OI + Delta Confirmed!\nStrike: {strike_price} | IV: {iv_change:.2f}% | OI: {oi_change:.2f}% | Delta: {delta:.2f}")

        # HIGH GAMMA ALERT
        if gamma > GAMMA_THRESHOLD:
            alerts.append(f"⚡ HIGH GAMMA: Big Move Incoming!\nStrike: {strike_price} | {opt_type}_Gamma: {gamma:.4f}")

        # HIGH TIME DECAY ALERT
        if theta < THETA_DECAY_THRESHOLD:
            alerts.append(f"⏳ HIGH TIME DECAY: Risk for Long Options!\nStrike: {strike_price} | {opt_type}_Theta: {theta:.2f}")

        # IV CRASH ALERT
        if iv_change < -IV_CRASH_THRESHOLD:
            alerts.append(f"🔥 IV CRASH ALERT: Sudden drop in IV!\nStrike: {strike_price} | {opt_type}_IV: {iv_change:.2f}%")

        # OI SURGE ALERT
        if oi_change > OI_SPIKE_THRESHOLD:
            alerts.append(f"🚀 OI SURGE ALERT: Institutional buying/selling!\nStrike: {strike_price} | {opt_type}_OI: {oi_change:.2f}%")

        # IV Rising but Price Stable Alert
        if iv_change > IV_SPIKE_THRESHOLD and price_change < PRICE_STABILITY_THRESHOLD:
            alerts.append(f"📈 IV RISING BUT PRICE STABLE: Expect Big Move Soon!\nStrike: {strike_price} | {opt_type}_IV: {iv_change:.2f}% | Price Change: {price_change:.2f}%")

        # SHORT SQUEEZE ALERT
        if iv_change > IV_SPIKE_THRESHOLD and oi_change > OI_SPIKE_THRESHOLD:
            alerts.append(f"🛑 SHORT SQUEEZE RISK: IV & OI surging together!\nStrike: {strike_price} | {opt_type}_IV: {iv_change:.2f}% | {opt_type}_OI: {oi_change:.2f}%")

        # CALLS DOMINATING
        if opt_type == "CE" and iv > df[df["Type"] == "PE"]["IV"].max():
            alerts.append(f"🟢 CALLS DOMINATING: Bullish sentiment detected!\nStrike: {strike_price} | CE_IV: {iv:.2f} (Change: {iv_change:.2f})")

        # PUTS DOMINATING
        if opt_type == "PE" and iv > df[df["Type"] == "CE"]["IV"].max():
            alerts.append(f"🔴 PUTS DOMINATING: Bearish sentiment detected!\nStrike: {strike_price} | PE_IV: {iv:.2f} (Change: {iv_change:.2f})")
        # STRADDLE TRIGGER ALERT
        if iv_change > IV_SPIKE_THRESHOLD and df[df["StrikePrice"] == strike_price]["IV"].max() > IV_SPIKE_THRESHOLD:
            alerts.append(f"💥 STRADDLE TRIGGER ALERT: Both CE & PE IV rising sharply!\nStrike: {strike_price}")

        # Directional Bias
        if iv_change > 5 and oi_change > 10 and delta > 0.75 and price_change > 0.5:
            alerts.append(f"📈 DIRECTIONAL BIAS: Bullish Setup!\nStrike: {strike_price}")
        elif iv_change > 5 and oi_change > 10 and delta < -0.75 and price_change < -0.5:
            alerts.append(f"📉 DIRECTIONAL BIAS: Bearish Setup!\nStrike: {strike_price}")

        # Reversal Setup Detection
        if iv_change < -IV_CRASH_THRESHOLD and oi_change > OI_SPIKE_THRESHOLD and abs(gamma) < 0.015:
            alerts.append(f"🌀 REVERSAL SETUP: IV drop with OI rise and low gamma!\nStrike: {strike_price}")

        # Save latest values
        previous_data[key] = {"IV": iv, "OI": oi}

    previous_data["underlying_price"] = underlying_price  # Save underlying price for next cycle

    if alerts:
        st.session_state.alerts = alerts + st.session_state.alerts[:10]  
        send_unique_telegram_alerts(alerts)

    save_to_csv(df)
    return df, net_oi_df


def send_unique_telegram_alerts(alerts):
    if "sent_alerts" not in st.session_state:
        st.session_state.sent_alerts = set()  # Use a set for efficiency

    new_alerts = [alert for alert in alerts if alert not in st.session_state.sent_alerts]

    if new_alerts:
        message = "\n".join(new_alerts)
        send_telegram_alert(message)  # Send only new alerts
        st.session_state.sent_alerts.update(new_alerts)  # Store sent alerts persistently

# Function to save data to CSV
def save_to_csv(df):
    if not os.path.exists(CSV_FILE):
        df.to_csv(CSV_FILE, index=False)
    else:
        df.to_csv(CSV_FILE, mode='a', header=False, index=False)

# 🔹 Main loop: Fetch and analyze every 3 minutes
def main():
    expiry_dates = get_expiry_dates()
    if not expiry_dates:
        st.error("No expiry dates found.")
        return

    nearest_expiry = expiry_dates[0]
    st.sidebar.write(f"**Nearest Expiry:** {nearest_expiry}")

    alert_placeholder = st.empty()  # Fixed placeholder for alerts
    table_placeholder = st.empty()  # Fixed placeholder for option chain data
    iv_chart_placeholder = st.empty()  # Fixed placeholder for IV chart
    oi_chart_placeholder = st.empty()  # Fixed placeholder for OI chart

    while True:
        st.sidebar.write("Fetching option chain data...")
        option_chain = fetch_option_chain(nearest_expiry)
        if option_chain:
            df, net_oi_df = analyze_data(option_chain)


            # 🔹 Display Dashboard Header
            st.title("📊 Nifty Options IV Spike Dashboard")

            # 🔹 Update Alerts Section
            with alert_placeholder.container():
                st.subheader("🔔 Real-time Alerts")
                for alert in st.session_state.alerts[:20]:
                    st.write(alert)

            # 🔹 Update Option Chain Data Table
            with table_placeholder.container():
                st.subheader("📜 ATM ± 4 Option Chain Data")
                st.dataframe(df)

            # 🔹 Update Net OI Imbalance Chart
            st.subheader("⚖️ Net OI Imbalance (CE - PE)")

            # Assign colors: Green for CE-dominant (NetOI > 0), Red for PE-dominant (NetOI < 0)
            net_oi_df['Color'] = net_oi_df['NetOI'].apply(lambda x: 'green' if x >= 0 else 'red')

            fig_netoi = px.bar(
                net_oi_df,
                x=net_oi_df.index,
                y="NetOI",
                title="Net OI per Strike",
                color='Color',
                color_discrete_map={'green': '#90ee90', 'red': '#ffcccb'}
            )
            st.plotly_chart(fig_netoi, use_container_width=True)

            # 🔹 Update IV & OI Charts
            col1, col2 = st.columns(2)
            with col1:
                with iv_chart_placeholder.container():
                    st.subheader("📈 IV Trend")
                    fig_iv = px.line(df, x="StrikePrice", y="IV", color="Type", title="IV vs Strike Price")
                    st.plotly_chart(fig_iv, use_container_width=True)

            with col2:
                with oi_chart_placeholder.container():
                    st.subheader("📊 OI Trend")
                    fig_oi = px.bar(
                        df,
                        x="StrikePrice",
                        y="OI",
                        color="Type",
                        title="OI vs Strike Price",
                        color_discrete_map={
                            "CE": "#90ee90",  # Light green
                            "PE": "#ffcccb"   # Light red
            }
                    )
                    st.plotly_chart(fig_oi, use_container_width=True)

        else:
            st.error("No data received.")

        # 🔹 Refresh Every 3 Minutes
        time.sleep(60)  # 3-minute delay
        st.rerun()  # Force rerun to update data

if __name__ == "__main__":
    main()
