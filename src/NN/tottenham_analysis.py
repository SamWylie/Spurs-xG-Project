import pandas as pd
import features
tottenham_df = pd.read_csv('data/shot_data.csv', delimiter=';')
tottenham_df = tottenham_df[
    ((tottenham_df['h_team'] == 'Tottenham') & (tottenham_df['h_a'] == 'h')) |
    ((tottenham_df['a_team'] == 'Tottenham') & (tottenham_df['h_a'] == 'a'))
]
tottenham_df.to_csv('data/processed_X.csv', index=False)
X, Y = features.trim(tottenham_df)
X.to_csv('data/tottenham_X.csv', index=False)

