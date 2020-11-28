import os
import pickle
import matplotlib.pyplot as plt

train_loss_file = os.path.join(os.getcwd(), 'dataset', 'twolayer-train-loss.pkl')
train_losses = None

with open(train_loss_file, 'rb') as f:
    train_losses = pickle.load(f)

plt.plot(train_losses)
plt.xlabel('Iterations')
plt.ylabel('loss')

plt.show()