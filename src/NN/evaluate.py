import torch
import model
from train import testX, testY, j, losses
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

model = model.Model()
model.load_state_dict(torch.load('models/final_model.pth'))
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


plt.plot(range(j+1), losses)
plt.ylabel("loss or error")
plt.xlabel('Epoch')
plt.show()