import streamlit as st
import numpy as np
from collections import Counter, defaultdict
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Hybrid AI Prediction", layout="wide")

st.title("🔮 Hybrid AI Prediction System (XGBoost + ARIMA + Markov + Patterns)")

# --- Input Section
default_data = "2,9,6,0,3,9,6,1,5,6,5,4,1,6,0,3,0,7,2,1,9,8,1,7,8,8,0,4,1,8,3,1,4,2,7,3,7,7,5,2,2,0,3,5,2,1,1,3,2,1"
user_input = st.text_area("📥 Enter last 50 results (comma-separated):", default_data)
past_results = list(map(int, user_input.strip().split(",")))

if len(past_results) < 50:
    st.warning("You need at least 50 values.")
    st.stop()

last_fifty = past_results[-50:]
X_train = np.array([past_results[i:i + 50] for i in range(len(past_results) - 50)])
y_train = np.array(past_results[50:])
if len(X_train.shape) == 1:
    X_train = X_train.reshape(-1, 50)

# --- Model Training
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.05, max_depth=3)
if len(X_train) > 0:
    xgb_model.fit(X_train, y_train)

stacking_model = StackingRegressor(
    estimators=[('xgb', xgb_model)],
    final_estimator=RandomForestRegressor(n_estimators=50),
    cv=5
)
if len(X_train) > 5:
    stacking_model.fit(X_train, y_train)

def build_markov_chain(data):
    matrix = np.zeros((10, 10))
    for i in range(len(data) - 1):
        matrix[data[i], data[i + 1]] += 1
    matrix = np.divide(matrix, matrix.sum(axis=1, keepdims=True), where=matrix.sum(axis=1, keepdims=True) != 0)
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

def get_predictions(last_fifty):
    return {
        "ML": int(round(xgb_model.predict([last_fifty])[0])),
        "Markov": markov_prediction(last_fifty),
        "ARIMA": arima_prediction(last_fifty),
    }

def hybrid_prediction(last_fifty):
    preds = get_predictions(last_fifty)
    model_input = np.array([[preds["ML"], preds["Markov"], preds["ARIMA"]]])
    final = int(round(stacking_model.final_estimator_.predict(model_input)[0]))
    return max(0, min(9, final)), preds

# --- Signal Functions
def classify(v): return "Small" if v <= 4 else "Big"
model_weights = {"ML": 0.5, "Markov": 0.3, "ARIMA": 0.2}

def majority_signal(preds):
    classes = [classify(p) for p in preds.values()]
    vote = Counter(classes).most_common(1)[0][0]
    return vote, Counter(classes)

def weighted_confidence_signal(preds, weights):
    score = sum(preds[m] * weights[m] for m in preds)
    avg = score / sum(weights.values())
    final_value = int(round(avg))
    return classify(final_value), final_value

def ai_signal_weighted_vote(preds, weights):
    class_weights = {"Small": 0.0, "Big": 0.0}
    for model, pred in preds.items():
        class_weights[classify(pred)] += weights.get(model, 0.0)
    return max(class_weights, key=class_weights.get), class_weights

def pattern_2_predict(data):
    count = defaultdict(lambda: defaultdict(int))
    for i in range(len(data) - 2):
        pattern = (data[i], data[i + 1])
        count[pattern][data[i + 2]] += 1
    last = tuple(data[-2:])
    return max(count[last], key=count[last].get) if last in count else np.random.randint(0, 10)

def pattern_3_predict(data):
    count = defaultdict(lambda: defaultdict(int))
    for i in range(len(data) - 3):
        pattern = (data[i], data[i + 1], data[i + 2])
        count[pattern][data[i + 3]] += 1
    last = tuple(data[-3:])
    return max(count[last], key=count[last].get) if last in count else np.random.randint(0, 10)

# --- Run Prediction
final_pred, model_preds = hybrid_prediction(last_fifty)
signal_majority, majority_votes = majority_signal(model_preds)
signal_weighted, weighted_val = weighted_confidence_signal(model_preds, model_weights)
signal_vote, vote_score = ai_signal_weighted_vote(model_preds, model_weights)
signal_final = classify(final_pred)
signal_ml = classify(model_preds["ML"])
pattern2 = pattern_2_predict(past_results)
pattern3 = pattern_3_predict(past_results)

# --- Output
st.subheader("📊 Model Predictions")
st.write(model_preds)

st.subheader("🎯 Final Prediction")
st.markdown(f"**{final_pred} ({signal_final})**")

st.subheader("🧠 Pattern-Based Predictions")
st.write({
    "Pattern2": f"{pattern2} ({classify(pattern2)})",
    "Pattern3": f"{pattern3} ({classify(pattern3)})"
})

st.subheader("📶 AI Signals")
st.write({
    "Majority Vote": (signal_majority, dict(majority_votes)),
    "Weighted Confidence": (signal_weighted, weighted_val),
    "Weighted Vote": (signal_vote, vote_score),
    "ML Signal": signal_ml
})

combined_output = f"{model_preds['ML']}{model_preds['Markov']}{model_preds['ARIMA']}{final_pred}{pattern2}{pattern3}"
binary_encoding = ''.join([
    '1' if signal_final == 'Big' else '0',
    '1' if signal_majority == 'Big' else '0',
    '1' if signal_weighted == 'Big' else '0',
    '1' if signal_vote == 'Big' else '0',
    '1' if signal_ml == 'Big' else '0',
])

st.subheader("🧾 Combined Output")
st.code(f"Combined: {combined_output} | Binary Signal: {binary_encoding}")
