from pytorch_workflow import plot_prediction
import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path


print(torch.__version__)

# setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")



# create some data using linear regression formula
weight = 0.7
bias = 0.3

# create range values
start = 0
end = 1
step = 0.02

# create X and Y (features and labels)
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# split the data
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(len(X_train), len(y_train), len(X_test), len(y_test))

# plt the data
# plot_prediction(X_train, y_train, X_test, y_test)


# build model
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        # create model parameter
        self.linear_layer = nn.Linear(in_features = 1 , out_features = 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)



# set manual seed
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
print(model_1, model_1.state_dict())




# check device
next(model_1.parameters()).device
# send model to target device
model_1.to(device)

# Trainig 
# 1. Loss function
# 2. Optimizer
# 3. Training loop
# 4. Testing loop




# set up loss function
loss_fn = nn.L1Loss()

# set up optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)

# write training loop
torch.manual_seed(42)
epochs = 200

for epoch in range(epochs):
    model_1.train()
    # forward pass
    y_pred = model_1(X_train)

    # calculate loss
    loss = loss_fn(y_pred, y_train)

    # optimizer zero grad
    optimizer.zero_grad()

    # perform back propagation
    loss.backward()

    # optimizer step
    optimizer.step()

    # Testing
    model_1.eval()

    with torch.inference_mode():
        test_pred = model_1(X_test)

        test_loss = loss_fn(test_pred, y_test)
    
    # print out
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss} | Tes loss: {test_loss}")

print(model_1.state_dict())


# Making predictions on the test data
model_1.eval()
with torch.inference_mode():
    y_preds = model_1(X_test)

print(y_preds)

# plot_prediction(predictions=y_preds.cpu())



# Saving our Pytorch Models
# created directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# create model save path
MODEL_NAME = "01_putting_all_together_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(MODEL_SAVE_PATH)

print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(), f=MODEL_SAVE_PATH)


loaded_model_1 = LinearRegressionModelV2()
print(loaded_model_1.load_state_dict(torch.load(f=MODEL_SAVE_PATH)))
print(loaded_model_1.state_dict())


loaded_model_1.eval()
with torch.inference_mode():
    loaded_model_1_preds = loaded_model_1(X_test)

print(y_preds == loaded_model_1_preds)