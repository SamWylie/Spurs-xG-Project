import model
import features
import torch
import torch.nn as nn
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)


epochs = 1000
losses = []
best_loss = float('inf')
patience = 20
counter = 0
for j in range(epochs):
    # go forward and get a prediction
    y_pred = model.forward(trainX) # get predicted results

    #measure the loss/erro, gonna be high at first
    loss = criterion(y_pred, trainY) # predicted values vs the y_train

    # keep track of our losses
    losses.append(loss.detach().numpy())

    #print every hundred run throughs
    # if j % 100 == 0:
    #     print(f'Epoch: {j} and loss: {loss}')

    # Do some back propagation: take the error rate of forward propagation and feed it nack through the network to fine tune the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if loss.item() < best_loss:
        best_loss = loss.item()
        counter = 0
        best_model_state = model.state_dict()
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {j}")
            model.load_state_dict(best_model_state)
            break




