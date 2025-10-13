import pickle
import pandas as pd

with open("xgb_manunited_model.pkl", "rb") as f:
    model=pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders=pickle.load(f)

with open("target_encoder.pkl", "rb") as f:
    target_encoder= pickle.load(f)

X_data = pd.DataFrame([[
    'Brighton', 'Newcastle Utd', 'The American Express Stadium', 'Chris Kavanagh',
    10, 5, 1, 1.6, 1.4, 1.78, 20, 0.6, 1.0, 0.74, 0
]], columns=['Home','Away', 'Venue', 'Referee','Month','DayOfWeek','IsWeekend',
              'home_goals_for_5', 'home_goals_against_5', 'home_xG_5', 'home_winrate_5',
              'away_goals_for_5', 'away_goals_against_5', 'away_xG_5', 'away_winrate_5'])


for col in ['Home', 'Away', 'Venue', 'Referee']:
    le = label_encoders[col]
    X_data[col] = le.transform(X_data[col])



predict= model.predict(X_data)
print("Outcome: ", target_encoder.inverse_transform(predict))
