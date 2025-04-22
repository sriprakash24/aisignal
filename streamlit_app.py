import streamlit as st
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA

# --- Initialize Past Results ---
past_results = [2,9,6,0,3,9,6,1,5,6,5,4,1,6,0,3,0,7,2,1,9,8,1,7,8,8,0,4,1,8,3,1,4,2,7,3,7,7,5,2,2,0,3,5,
2,1,1,3,2,1]

# --- Define Helper Functions ---
def classify(v): return "Small" if v <= 4 else "Big"

def build_markov_chain(data):
    size = 10
    matrix = np.zeros((size, size))
    for i in range(len(data) - 1):
        matrix[data[i], data[i + 1]] += 1
    for row in range(size):
        total = np.sum(matrix[row])
        if total > 0:
            matrix[row] /= total
    return matrix

def markov_prediction(data):
    last = data[-1]
    row = transition_matrix[last]
    return int(np.argmax(row)) if np.sum(row) > 0 else np.random.randint(0, 10)

def arima_prediction(data):
    try:
        if len(data) < 3: return np.random.randint(0, 10)
        model = ARIMA(data, order=(1, 1, 1))
        model_fit = model.fit()
        pred = int(round(model_fit.forecast()[0]))
        return max(0, min(9, pred))
    except:
        return np.random.randint(0, 10)

def pattern_2_predict(data):
    pattern_counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(data) - 2):
        pattern = (data[i], data[i + 1])
        next_val = data[i + 2]
        pattern_counts[pattern][next_val] += 1
    last_pattern = tuple(data[-2:])
    return max(pattern_counts[last_pattern], key=pattern_counts[last_pattern].get) if last_pattern in pattern_counts else np.random.randint(0, 10)

def pattern_3_predict(data):
    pattern_counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(data) - 3):
        pattern = (data[i], data[i + 1], data[i + 2])
        next_val = data[i + 3]
        pattern_counts[pattern][next_val] += 1
    last_pattern = tuple(data[-3:])
    return max(pattern_counts[last_pattern], key=pattern_counts[last_pattern].get) if last_pattern in pattern_counts else np.random.randint(0, 10)

# --- Train Models ---
X_train = np.array([past_results[i:i+50] for i in range(len(past_results)-50)])
y_train = np.array(past_results[50:])
X_train = X_train.reshape(-1, 50)

xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.05, max_depth=3)
if len(X_train) > 0:
    xgb_model.fit(X_train, y_train)

transition_matrix = build_markov_chain(past_results)

stacking_model = StackingRegressor(
    estimators=[('xgb', xgb_model)],
    final_estimator=RandomForestRegressor(n_estimators=50),
    cv=5
)
if len(X_train) > 5:
    stacking_model.fit(X_train, y_train)

# --- Short-Term Stacking ---
short_X, short_y = [], []
for i in range(len(past_results) - 5):
    seq = past_results[i:i+5]
    short_X.append([markov_prediction(seq), pattern_2_predict(seq), pattern_3_predict(seq)])
    short_y.append(past_results[i+5])
short_X = np.array(short_X)
short_y = np.array(short_y)

short_stacking_model = StackingRegressor(
    estimators=[
        ('rf', RandomForestRegressor(n_estimators=50)),
        ('xgb', XGBRegressor(objective='reg:squarederror', n_estimators=50))
    ],
    final_estimator=RandomForestRegressor(n_estimators=30),
    cv=3
)
if len(short_X) > 5:
    short_stacking_model.fit(short_X, short_y)

# --- Prediction and Visualization ---
st.title("🔮 Hybrid AI Prediction Dashboard")

last_50 = past_results[-50:]
preds = {
    "ML": int(round(xgb_model.predict([last_50])[0])),
    "Markov": markov_prediction(last_50),
    "ARIMA": arima_prediction(last_50)
}
final_pred = int(round(stacking_model.final_estimator_.predict(np.array([[preds[k] for k in preds]]))[0]))
pattern2 = pattern_2_predict(past_results)
pattern3 = pattern_3_predict(past_results)
short_term_result = int(round(short_stacking_model.predict(np.array([[preds['Markov'], pattern2, pattern3]]))[0]))

binary_encoding = ''.join(['1' if classify(preds[k]) == 'Big' else '0' for k in preds])
binary_encoding += '1' if classify(final_pred) == 'Big' else '0'
binary_encoding += '1' if classify(short_term_result) == 'Big' else '0'

# --- Display Results ---
st.subheader("📈 Predictions")
st.write(pd.DataFrame({
    'Model': list(preds.keys()) + ['Final Stacked', 'Pattern2', 'Pattern3', 'Short-Term Stacked'],
    'Prediction': list(preds.values()) + [final_pred, pattern2, pattern3, short_term_result],
    'Group': [classify(p) for p in list(preds.values()) + [final_pred, pattern2, pattern3, short_term_result]]
}))

st.subheader("🧠 Binary Signal Encoding")
st.code(binary_encoding)

st.markdown("---")
st.write("📊 This dashboard combines multiple AI models including XGBoost, ARIMA, Markov Chain, Pattern Recognition, and Stacking for both long-term and short-term predictions.")
