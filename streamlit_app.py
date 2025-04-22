import streamlit as st
import numpy as np
from collections import Counter, defaultdict
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.utils.validation import check_is_fitted

# Title
st.title("🔮 AI Hybrid Prediction Dashboard")

# Past results (editable or hardcoded for now)
past_results = [2,9,6,0,3,9,6,1,5,6,5,4,1,6,0,3,0,7,2,1,9,8,1,7,8,8,0,4,1,8,3,1,4,2,7,3,7,7,5,2,2,0,3,5,2,1,1,3,2,1]

if len(past_results) < 50:
    st.error("Need at least 50 past results.")
    st.stop()

X_train = np.array([past_results[i:i+50] for i in range(len(past_results)-50)])
y_train = np.array(past_results[50:])

if len(X_train.shape) == 1:
    X_train = X_train.reshape(-1, 50)

# ML Model
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.05, max_depth=3)
if len(X_train) > 0:
    xgb_model.fit(X_train, y_train)

# Stacking model
stacking_model = StackingRegressor(
    estimators=[('xgb', xgb_model)],
    final_estimator=RandomForestRegressor(n_estimators=50),
    cv=5
)
if len(X_train) > 5:
    stacking_model.fit(X_train, y_train)

# Markov Chain
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

transition_matrix = build_markov_chain(past_results)

# Predictors
def markov_prediction(data):
    last = data[-1]
    if np.sum(transition_matrix[last]) == 0:
        return np.random.randint(0, 10)
    return int(np.argmax(transition_matrix[last]))

def arima_prediction(data):
    try:
        model = ARIMA(data, order=(1,1,1))
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
    if last_pattern in pattern_counts:
        return max(pattern_counts[last_pattern], key=pattern_counts[last_pattern].get)
    return np.random.randint(0, 10)

def pattern_3_predict(data):
    pattern_counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(data) - 3):
        pattern = (data[i], data[i + 1], data[i + 2])
        next_val = data[i + 3]
        pattern_counts[pattern][next_val] += 1
    last_pattern = tuple(data[-3:])
    if last_pattern in pattern_counts:
        return max(pattern_counts[last_pattern], key=pattern_counts[last_pattern].get)
    return np.random.randint(0, 10)

def classify(v):
    return "Small" if v <= 4 else "Big"

# Last 50
last_50 = past_results[-50:]

# Safe ML prediction
try:
    check_is_fitted(xgb_model)
    ml_pred = int(round(xgb_model.predict([last_50])[0]))
except:
    ml_pred = np.random.randint(0, 10)

# Model predictions
model_preds = {
    "ML": ml_pred,
    "Markov": markov_prediction(last_50),
    "ARIMA": arima_prediction(last_50),
}

# Final stacking prediction
try:
    stacked_input = np.array([[model_preds['ML'], model_preds['Markov'], model_preds['ARIMA']]])
    final_pred = int(round(stacking_model.final_estimator_.predict(stacked_input)[0]))
except:
    final_pred = np.random.randint(0, 10)

# Signals
signal_majority = Counter([classify(v) for v in model_preds.values()]).most_common(1)[0][0]
model_weights = {"ML": 0.5, "Markov": 0.3, "ARIMA": 0.2}
weighted_score = sum(model_preds[m] * model_weights[m] for m in model_preds)
avg_score = weighted_score / sum(model_weights.values())
signal_weighted = "Big" if avg_score > 4 else "Small"
signal_final = classify(final_pred)
signal_ml = classify(ml_pred)
class_weights = {"Small": 0.0, "Big": 0.0}
for m, pred in model_preds.items():
    class_weights[classify(pred)] += model_weights.get(m, 0.0)
signal_weighted_vote = max(class_weights, key=class_weights.get)

# Pattern results
pattern2_result = pattern_2_predict(past_results)
pattern3_result = pattern_3_predict(past_results)

# Binary encoding
binary_encoding = ''.join([
    '1' if signal_final == 'Big' else '0',
    '1' if signal_majority == 'Big' else '0',
    '1' if signal_weighted == 'Big' else '0',
    '1' if signal_weighted_vote == 'Big' else '0',
    '1' if signal_ml == 'Big' else '0'
])

combined_output = f"{ml_pred}{model_preds['Markov']}{model_preds['ARIMA']}{final_pred}{pattern2_result}{pattern3_result}"

# Display
st.subheader("📊 Model Predictions")
st.write(model_preds)
st.write(f"Final Prediction (Stacking): {final_pred} ({signal_final})")

st.subheader("📈 Pattern-Based Predictions")
st.write(f"Pattern2: {pattern2_result} ({classify(pattern2_result)})")
st.write(f"Pattern3: {pattern3_result} ({classify(pattern3_result)})")

st.subheader("🔢 Binary Encoding")
st.code(binary_encoding)

st.subheader("🔗 Combined Output")
st.code(combined_output)
