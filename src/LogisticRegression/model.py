import features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

X = features.df.drop('result', axis=1)
Y = features.df['result']

trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(trainX, trainY)

predictions = model.predict_proba(testX)[:, 1]
print(predictions[0:10])

final_df = testX.copy()
final_df['goal_probability'] = predictions

final_df.iloc[830]
print(final_df.sort_values(by='goal_probability', ascending=False).head())

logloss = log_loss(testY, predictions)
roc_auc = roc_auc_score(testY, predictions)
brier_score = brier_score_loss(testY, predictions)

print(f"Log Loss: {logloss}")
print(f"ROC AUC: {roc_auc}")
print(f"Brier Score: {brier_score}")
