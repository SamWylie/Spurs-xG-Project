import data_loader
import pandas as pd
import numpy as np

my_df = data_loader.my_df


my_df = my_df[(my_df['X'] <= 1) & (my_df['Y'] <= 1)]
my_df = my_df[my_df['minute'] <= 90]
my_df = my_df[my_df['result'] != 'Own Goal']

# Map 'Goal' to 1.0 and all other results to 0.0
if 'result' in my_df.columns:
    my_df['result'] = my_df['result'].apply(lambda x: 1.0 if x == 'Goal' else 0.0)

def add_distance_angle(df, x_col="X", y_col="Y"):
    """
    Adds distance and angle features to a dataframe of shots.

    Assumes pitch is normalized (X and Y between 0 and 1).
    Goal center is at (1.0, 0.5).
    Goalposts are at (1.0, 0.44) and (1.0, 0.56).
    """
    goal_x = 1.0
    goal_y_left = 0.44
    goal_y_right = 0.56
    goal_center_y = (goal_y_left + goal_y_right) / 2

    distances = []
    angles = []

    for _, row in df.iterrows():
        x, y = row[x_col], row[y_col]

        # Distance to goal center
        d = np.sqrt((goal_x - x) ** 2 + (goal_center_y - y) ** 2)

        # Angle as "view of goal mouth"
        left_post = np.arctan2(goal_y_left - y, goal_x - x)
        right_post = np.arctan2(goal_y_right - y, goal_x - x)
        a = abs(right_post - left_post)

        distances.append(d)
        angles.append(a)

    df["distance"] = distances
    df["angle"] = angles

    return df



# Example: apply to your shots dataframe
my_df = add_distance_angle(my_df, x_col="X", y_col="Y")

# Drop raw coords if you don’t want them
my_df = my_df.drop(columns=["X", "Y"])

#dropping uneeded training columns
X = my_df.drop(columns=['id', 'minute', 'h_goals', 'a_goals', 'lastAction','result', 'player_id', 'season', 'xG', 'h_team', 'a_team', 'player', 'match_id', 'date', 'player_assisted'])
X = X.drop(columns=[col for col in X.columns if col.startswith('Unnamed')])

#result column
Y = my_df['result']











# Rename 'h_a' to 'h' and map values
if 'h_a' in X.columns:
    X = X.rename(columns={'h_a': 'h'})
    X['h'] = X['h'].map({'h': 1.0, 'a': 0.0})

# --- Neural Network Data Preparation Steps ---

# 2. Remove or impute any missing values (NaN)
X = X.fillna(0)  

# 3. Scale/normalize features so they are on a similar range

# One-hot encode categorical columns before scaling
categorical_cols = [col for col in ['situation', 'shotType'] if col in X.columns]
if categorical_cols:
    X = pd.get_dummies(X, columns=categorical_cols)

# Convert boolean columns to float
for col in X.columns:
    if X[col].dtype == bool:
        X[col] = X[col].astype(float)

# 4. Make sure your target (Y) is float and only 0.0 or 1.0 for binary classification
Y = Y.astype(float)
Y = Y.clip(0, 1) 

print(X.head())

X.to_csv('data/processed_X.csv', index=False)
Y.to_csv('data/processed_Y.csv', index=False)

