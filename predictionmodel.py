import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
import xgboost as xgb
from sklearn.metrics import accuracy_score

data = pd.read_csv("C:/Users/swastik limbu/Desktop/manunited.csv", encoding='utf-8-sig')
data.columns = data.columns.str.strip()

data['Date']=pd.to_datetime(data['Date'], errors='coerce')
data[['HomeGoals','AwayGoals']]= data['Score'].str.extract(r'(\d+)–(\d+)').astype(float)

def result(row):
    if row['HomeGoals']>row['AwayGoals']:
        return 'HomeWin'
    elif row['HomeGoals']<row['AwayGoals']:
        return 'AwayWin'
    else:
        return 'Draw'

data['Result']=data.apply(result, axis=1)

def compute_team_form(df, team_col, gf_col, ga_col, xg_col, prefix):
    team_stats= []
    for team,tdf in df.groupby(team_col, sort=False):
        tdf = tdf.sort_values('Date')
        tdf[f'{prefix}_goals_for_5']= tdf[gf_col].shift().rolling(5,min_periods=1).mean()
        tdf[f'{prefix}_goals_against_5']= tdf[ga_col].shift().rolling(5,min_periods=1).mean()
        tdf[f'{prefix}_xG_5']= tdf[xg_col].shift().rolling(5,min_periods=1).mean()

        tdf[f'{prefix}_winrate_5']=(
            ((tdf['Result'] == 'HomeWin') if prefix =='home' else (tdf['Result'] == 'AwayWin') )
            .astype(int).shift().rolling(5).mean()
        )
        team_stats.append(tdf)
    
    return pd.concat(team_stats)

home_stats= compute_team_form(data,'Home', 'HomeGoals','AwayGoals', 'xG', 'home')
away_stats= compute_team_form(data, 'Away', 'AwayGoals', 'HomeGoals','xG.1', 'away')

data= home_stats.copy()
data[['away_goals_for_5','away_goals_against_5','away_xG_5','away_winrate_5']] = (
    away_stats[['away_goals_for_5','away_goals_against_5','away_xG_5','away_winrate_5']]
)

data['Month']= data['Date'].dt.month
data['DayOfWeek']= data['Date'].dt.dayofweek
data['IsWeekend']= data['DayOfWeek'].isin([5,6]).astype(int)

label_cols= [ 'Home', 'Away','Venue', 'Referee']
encoders= {}
for col in label_cols:
    enc= LabelEncoder()
    data[col]=enc.fit_transform(data[col])
    encoders[col] = enc


features= ['Home','Away', 'Venue', 'Referee','Month','DayOfWeek','IsWeekend', 'home_goals_for_5', 'home_goals_against_5', 'home_xG_5', 'home_winrate_5',
    'away_goals_for_5', 'away_goals_against_5', 'away_xG_5', 'away_winrate_5']

target= 'Result'

model_data= data.dropna(subset=features + [target])

X= model_data[features]
y= model_data[target]

print(model_data['Result'].value_counts())

le=LabelEncoder()
y_fin= le.fit_transform(y)


x_train, x_test, y_train, y_test= train_test_split(X,y_fin, test_size=0.2, random_state=42)

param_grid= {
    'n_estimators': [200,300],
    'max_depth': [3,5],
    'learning_rate': [0.01,0.02],
    'subsample': [0.6,0.8],
    'colsample_bytree': [0.6,0.8],
}

model= xgb.XGBClassifier(eval_metric='logloss')

grid= RandomizedSearchCV(
    model,
    param_grid,
    n_iter=20,
    cv= 3,
    n_jobs=-1,
    scoring='accuracy',
    random_state=42
)

grid.fit(x_train,y_train)

best_model= grid.best_estimator_

scoring=cross_val_score(best_model,X,y_fin, cv=4, scoring="accuracy")
print("Cross performance: ",scoring.mean())

predict= grid.predict(x_test)
print("Accuracy: ",accuracy_score(y_test,predict))

with open("xgb_manunited_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

with open("target_encoder.pkl", "wb") as f:
    pickle.dump(le, f)