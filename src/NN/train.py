import model
import features
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

torch.manual_seed(42)
model = model.Model()
X = features.X.values
Y = features.Y.values


from sklearn.model_selection import train_test_split
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=42)

trainX = torch.FloatTensor(trainX)
trainY = torch.FloatTensor(trainY).unsqueeze(1)
testX = torch.FloatTensor(testX)
testY = torch.FloatTensor(testY).unsqueeze(1)

criterion = nn.BCEWithLogitsLoss() #binary cross entropy loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 1000
losses = []
for j in range(epochs):
    # go forward and get a prediction
    y_pred = model.forward(trainX) # get predicted results

    #measure the loss/erro, gonna be high at first
    loss = criterion(y_pred, trainY) # predicted values vs the y_train

    # keep track of our losses
    losses.append(loss.detach().numpy())

    #print every hundred run throughs
    if j % 100 == 0:
        print(f'Epoch: {j} and loss: {loss}')

    # Do some back propagation: take the error rate of forward propagation and feed it nack through the network to fine tune the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
model.eval()
with torch.no_grad():
    test_logits = model(testX)
    test_probs = torch.sigmoid(test_logits).numpy().flatten()
    testY_np = testY.numpy().flatten()


ll = log_loss(testY_np, test_probs)
roc_auc = roc_auc_score(testY_np, test_probs)
brier = brier_score_loss(testY_np, test_probs)

print(f'Log Loss: {ll:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')
print(f'Brier Score: {brier:.4f}')



