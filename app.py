from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import xgboost as xgb
import json

# Initialize Flask app
app = Flask(__name__)

# Load model and feature list
model = xgb.Booster()
model.load_model("xgboost_model_devcamp.json")

with open("features.json", "r") as f:
    FEATURES = json.load(f)

# Label encoders must match training
from sklearn.preprocessing import LabelEncoder

# Dummy encoders just for structure – in production, load fitted encoders from disk
store_enc = LabelEncoder()
sku_enc = LabelEncoder()
store_enc.classes_ = np.load("store_classes.npy", allow_pickle=True)
sku_enc.classes_ = np.load("sku_classes.npy", allow_pickle=True)

# Utility: Feature engineering (must match training logic)
def feature_engineering(df):
    df['discount'] = (df['base_price'] - df['total_price']).clip(lower=0)
    df['discount_pct'] = df['discount'] / df['base_price']
    df['week'] = pd.to_datetime(df['week'])
    df['week_number'] = df['week'].dt.isocalendar().week
    df['year'] = df['week'].dt.year
    df['price_rank_in_store_week'] = df.groupby(['store_id', 'week'])['total_price'].rank()
    return df

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if isinstance(data, dict):
            data = [data]  # handle single input

        df = pd.DataFrame(data)

        # Feature engineering
        df = feature_engineering(df)

        # Encode categories
        df['store_id_enc'] = store_enc.transform(df['store_id'])
        df['sku_id_enc'] = sku_enc.transform(df['sku_id'])

        # Ensure feature order and extract input
        X = df[FEATURES]
        dmatrix = xgb.DMatrix(X)

        # Predict
        preds = model.predict(dmatrix)
        preds_rounded = np.round(np.clip(preds, 0, None)).astype(int)

        return jsonify({"predictions": preds_rounded.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
