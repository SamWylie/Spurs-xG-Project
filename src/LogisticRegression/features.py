import data_loader
import pandas as pd
import numpy as np

df = data_loader.my_df
df = df[df['result'] != 'Own Goal']
df = df[(df['X'] <= 1) & (df['Y'] <= 1)]
df['result'] = df['result'].apply(lambda x: True if x == 'Goal' else False)
df = df.drop(columns=['id', 'h_a', 'date', 'minute', 'h_goals', 'a_goals', 'player_id', 'season', 'xG', 'h_team', 'a_team', 'player', 'match_id', 'player_assisted'])
df = df.drop(columns=[col for col in df.columns if col.startswith('Unnamed')])
categorical_cols = [col for col in ['situation', 'shotType', 'lastAction'] if col in df.columns]
if categorical_cols:
    df = pd.get_dummies(df, columns=categorical_cols)

df["distance"] = np.sqrt((df['X'] - 1)**2 + (df['Y'] - 0.5)**2)
print(df.info())
print(df["distance"].describe())