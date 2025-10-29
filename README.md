<h1 align="center">⚽ Premier League Match Outcome Predictor</h1>

<p align="center">
  A machine learning model that predicts Premier League match results — Home Win, Away Win, or Draw.
</p>

---

<h2>📘 Overview</h2>

This project uses past match data to predict the outcome of future games.  
It applies basic preprocessing, form-based features, and an XGBoost classifier for accurate predictions.

---

<h2>🧠 Tech Stack</h2>

- Python (Pandas, NumPy)
- Scikit-learn  
- XGBoost  
- Pickle  

---

<h2>📊 Dataset</h2>

The dataset (<code>premier.csv</code>) includes:
- Date, Home, Away, Score, Venue, Referee, and Expected Goals (xG)

---

<h2>⚙️ Process</h2>

1. Data cleaning and conversion of date and score columns  
2. Creation of match result labels (HomeWin / AwayWin / Draw)  
3. Rolling averages for last 5 matches (goals, xG, win rate)  
4. Encoding of categorical features  
5. Training XGBoost classifier with RandomizedSearchCV  
6. Evaluation using accuracy and cross-validation  

---

<h2>📁 Output Files</h2>

| File | Description |
|------|--------------|
| `xgb_manunited_model.pkl` | Trained XGBoost model |
| `label_encoders.pkl` | Encoders for categorical features |
| `target_encoder.pkl` | Encoder for target variable |

---

<h2>🚀 Run Instructions</h2>

```bash
git clone https://github.com/BisheshSubba/Premier-league-Prediction-model.git
cd Premier-league-Prediction-model
pip install -r requirements.txt
python main.py
