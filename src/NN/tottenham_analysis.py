import pandas as pd
import features
import numpy as np
import torch
import model
tottenham_df = pd.read_csv('data/shot_data.csv', delimiter=';')
tottenham_df = tottenham_df[
    ((tottenham_df['h_team'] == 'Tottenham') & (tottenham_df['h_a'] == 'h')) |
    ((tottenham_df['a_team'] == 'Tottenham') & (tottenham_df['h_a'] == 'a'))
]
X, Y = features.trim(tottenham_df)
X.to_csv('data/tottenham_X.csv', index=False)

spurs_model = model.Model(in_features=X.shape[1])
spurs_model.load_state_dict(torch.load('models/final_model.pth'))
spurs_model.eval()

with torch.no_grad():
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    Y_pred = spurs_model(X_tensor)

Y_pred = torch.sigmoid(Y_pred).numpy().flatten()


analysis_df = pd.DataFrame()
analysis_df['result'] = Y
analysis_df['xG'] = Y_pred
analysis_df['player'] = tottenham_df['player']
analysis_df['year'] = tottenham_df['season']
analysis_df.to_csv('data/analysis.csv', index=False)

def years_xG_goals(analysis_df):
    # Group by 'years' and sum xG + goals
    result = (
        analysis_df.groupby('year')[['xG', 'result']]
        .sum()
        .reset_index()
    )
    return result


def player_xG_vs_goals(analysis_df):
    temp = analysis_df[analysis_df['year'] >= 2000]
    result = (
        temp.groupby('player')[['xG', 'result']]
        .sum()
        .reset_index()
    )
    return result

result1 = years_xG_goals(analysis_df)
result2 = player_xG_vs_goals(analysis_df)









