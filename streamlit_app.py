import streamlit as st
import numpy as np
from collections import Counter, defaultdict
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA

# 🔹 Streamlit Title
st.title("📊 Hybrid AI Prediction App")

# 🔹 Input for Past Results
user_input = st.text_area("Enter at least 51 past results (comma-separated):", 
                          "2,9,6,0,3,9,6,1,5,6,5,4,1,6,0,3,0,7,2,1,9,8,1,7,8,8,0,4,1,8,3,1,4,2,7,3,7,7,5,2,2,0,3,5,2,1,1,3,2,1,4")

# 🔹 Convert input to list
try:
    past_results = [int(x.strip()) for x in user_input.split(",") if x.strip().isdigit()]
except:
    st.error("❌ Invalid input. Please enter only comma-separated numbers.")
    st.stop()

# 🔹 Check length
if len(past_results) < 51:
    st.warning(f"⚠️ Need at least 51 numbers. You provided {len(past_results)}.")
    st.stop()
