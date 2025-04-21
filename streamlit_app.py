import numpy as np
import streamlit as st
from collections import Counter, defaultdict
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA

st.title("🔮 Hybrid AI Prediction System")
st.markdown("This app uses XGBoost, ARIMA, Markov Chains, and Pattern Matching to predict the next value from past results.")

# 🔹 Updated Last 50 Results (replace this with your dynamic input if needed)
past_results = [2,9,6,0,3,9,6,1,5,6,5,4,1,6,0,3,0,7,2,1,9,8,1,7,8,8,0,4,1,8,3,1,4,2,7,3,7,7,5,2,2,0,3,5,2,1,1,3,2,1]

if len(past_results) < 51:
    st.error("Need at least 51 past results for training.")
    st.stop()

# 🔹 Prepare Data
X_train = np.array([past_results[i:i+50] for i in range(len(past_results)-50)])
y_train = np.array(past_results[50:])

# 🔹 XGBoost Model
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.05, max_depth=3)
if len(X_train) > 0:
    xgb_model.fit(X_train, y_train)

# 🔹 Markov Chain
def build_markov_chain(data):
    matrix = np.zeros((10, 10))
    for i in range(len(data) - 1):
        matrix[data[i], data[i + 1]] += 1
    matrix = np.divide(matrix, matrix.sum(axis=1, keepdims=True), where=matrix.sum(axis=1, keepdims=True)!=0)
    return matrix

transition_matrix = build_markov_chain(past_results)

def markov_prediction(last_fifty):
    last_index = last_fifty[-1]
    if np.sum(transition_matrix[last_index]) == 0:
        return np.random.randint(0, 10)
    return int(np.argmax(transition_matrix[last_index]))

def arima_prediction(last_fifty):
    try:
        model = ARIMA(last_fifty, order=(1, 1, 1))
        model_fit = model.fit()
        pred = int(round(model_fit.forecast()[0]))
        return max(0, min(9, pred))
    except:
        return np.random.randint(0, 10)

# 🔹 Stacking Model
stacking_model = StackingRegressor(
    estimators=[('xgb', xgb_model)],
    final_estimator=RandomForestRegressor(n_estimators=50),
    cv=5
)
if len(X_train) > 5:
    stacking_model.fit(X_train, y_train)

# 🔹 Prediction Logic
def get_predictions(last_fifty):
    if not hasattr(xgb_model, "feature_importances_"):
        return {
            "ML": np.random.randint(0, 10),
            "Markov": markov_prediction(last_fifty),
            "ARIMA": arima_prediction(last_fifty),
        }
    return {
        "ML": int(round(xgb_model.predict([last_fifty])[0])),
        "Markov": markov_prediction(last_fifty),
        "ARIMA": arima_prediction(last_fifty),
    }

def hybrid_prediction(last_fifty):
    preds = get_predictions(last_fifty)
    model_inputs = np.array([[preds["ML"], preds["Markov"], preds["ARIMA"]]])
    if hasattr(stacking_model.final_estimator_, "predict"):
        final_pred = int(round(stacking_model.final_estimator_.predict(model_inputs)[0]))
    else:
        final_pred = np.random.randint(0, 10)
    return max(0, min(9, final_pred)), preds

# 🔹 Signals
def classify(v): return "Small" if v <= 4 else "Big"

def majority_signal(predictions):
    classes = [classify(p) for p in predictions.values()]
    vote = Counter(classes).most_common(1)[0][0]
    return vote, Counter(classes)

def weighted_confidence_signal(predictions, weights):
    weighted = sum(predictions[m] * weights[m] for m in predictions)
    avg = weighted / sum(weights.values())
    return classify(int(round(avg))), int(round(avg))

def ai_signal_weighted_vote(predictions, weights):
    scores = {"Small": 0.0, "Big": 0.0}
    for model, val in predictions.items():
        scores[classify(val)] += weights.get(model, 0.0)
    return max(scores, key=scores.get), scores

# 🔹 Pattern Logic
def pattern_2_predict(data):
    pattern_counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(data) - 2):
        pattern_counts[(data[i], data[i + 1])][data[i + 2]] += 1
    last_pattern = tuple(data[-2:])
    if last_pattern in pattern_counts:
        return max(pattern_counts[last_pattern], key=pattern_counts[last_pattern].get)
    return np.random.randint(0, 10)

def pattern_3_predict(data):
    pattern_counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(data) - 3):
        pattern_counts[(data[i], data[i + 1], data[i + 2])][data[i + 3]] += 1
    last_pattern = tuple(data[-3:])
    if last_pattern in pattern_counts:
        return max(pattern_counts[last_pattern], key=pattern_counts[last_pattern].get)
    return np.random.randint(0, 10)

# 🔹 Run Prediction
last_fifty = past_results[-50:]
final_pred, model_preds = hybrid_prediction(last_fifty)

pattern2 = pattern_2_predict(past_results)
pattern3 = pattern_3_predict(past_results)

# 🔹 Display
st.subheader("🔢 Model Predictions")
st.write(model_preds)

st.subheader("🎯 Final Prediction (Stacking)")
st.write(f"{final_pred} ({classify(final_pred)})")

st.subheader("🧠 Pattern Predictions")
st.write(f"Pattern 2: {pattern2} ({classify(pattern2)})")
st.write(f"Pattern 3: {pattern3} ({classify(pattern3)})")

model_weights = {"ML": 0.5, "Markov": 0.3, "ARIMA": 0.2}

signal_majority, _ = majority_signal(model_preds)
signal_weighted, score_weighted = weighted_confidence_signal(model_preds, model_weights)
signal_vote, vote_scores = ai_signal_weighted_vote(model_preds, model_weights)

st.subheader("📶 AI Signals")
st.write(f"Majority Signal: {signal_majority}")
st.write(f"Weighted Confidence Signal: {signal_weighted} (Avg Score: {score_weighted})")
st.write(f"Weighted Vote Signal: {signal_vote} (Weights: {vote_scores})")

# 🔹 Output Encoding
binary = ''.join([
    '1' if classify(final_pred) == 'Big' else '0',
    '1' if signal_majority == 'Big' else '0',
    '1' if signal_weighted == 'Big' else '0',
    '1' if signal_vote == 'Big' else '0',
    '1' if classify(model_preds["ML"]) == 'Big' else '0',
])
combined_output = f"{model_preds['ML']}{model_preds['Markov']}{model_preds['ARIMA']}{final_pred}{pattern2}{pattern3}"

st.subheader("🧮 Combined Outputs")
st.write(f"Binary Signal Encoding: `{binary}`")
st.write(f"Combined Output: `{combined_output}`")
