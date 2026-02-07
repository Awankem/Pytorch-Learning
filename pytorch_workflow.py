# Removed invalid import
from sys import modules
import torch
from torch import nn #contains all pytorch's buiding blocks for nueral networs
import matplotlib.pyplot as plt


print(torch.__version__)

# create know parameter
weight = 0.7
bias = 0.3

# create
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

print(X[:10], y[:10])
print(len(X), len(y))




# Splitting data into training and test sets
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(len(X_train), len(y_train), len(X_test), len(y_test))



# Visualization
def plot_prediction(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=None):

    # predictions should be passed as parameter or generated before calling this function

    plt.figure(figsize=(10,7))
    # training data
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    # test data
    plt.scatter(test_data, test_labels, c="y", s=4, label="Test data")


    # Are there predictions
    if predictions is not None:
        # plot predictions
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    
    # show legend and display plot
    plt.legend(prop={"size":14})
    plt.show()

# plot_prediction();


# Building our first Pytorch Model
# create a linear regression
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Use simple parameters for linear regression (y = mx + c)
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


torch.manual_seed(42)
model_0 = LinearRegressionModel()

# Setup loss function and optimizer
loss_fn = nn.L1Loss() # MAE loss
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

# Training loop
epochs = 200

for epoch in range(epochs):
    model_0.train()
    y_pred = model_0(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch: {epoch} | Loss: {loss}")

# Save the model
MODEL_PATH = "model.pth"
torch.save(model_0.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# Make predictions with the trained model
model_0.eval()
with torch.inference_mode():
    y_preds = model_0(X_test)

print(y_preds)
# plot_prediction(predictions=y_preds) # Commented out for automated environment


# loss function
# a loss function measures how wrong the model's predictions are
# torch.nn.L1Loss
# torch.nn.MSELoss

# optimizer
# an optimizer updates the model's parameters to reduce the loss
# torch.optim.SGD
# torch.optim.Adam
