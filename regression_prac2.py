import numpy as np
import pandas as pd
import torch 
from torch import nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class LinearRegressionModel(nn.Module):

    def __init__(self, len_x_labels):

        super().__init__()
        self.weights = nn.Parameter(torch.randn(len_x_labels, 1), requires_grad = True)
        self.bias = nn.Parameter(torch.randn(1), requires_grad = True)

    def forward(self, x):

        return torch.matmul(x, self.weights) + self.bias

def get_xlabels(df, y_label, threshold):

    corr_df = df.corr(numeric_only = True)
    new_dict = {}
    x_labels = []

    for column in corr_df.columns:

        if column == y_label:
            continue

        new_dict[column] = corr_df[column][y_label]

    for feature, correlation in new_dict.items():

        if abs(correlation) > threshold:

            x_labels.append(feature)

    return x_labels

def get_xy(df, y_label, x_labels = None):

    X = -1   #placeholder value

    if x_labels == None:
        X = df[[c for c in df.columns if c != y_label]].values
    elif len(x_labels) == 1:
        X = df[x_labels[0]].values.reshape(-1, 1)
    else:
        X = df[x_labels].values

    y = df[y_label].values.reshape(-1, 1)

    return X, y

def train_loop(model, X_train, y_train, X_validate, y_validate, loss_fn, optimizer, epochs):

    training_losses, validation_losses = [], []

    for epoch in range(epochs):

        model.train()
    
        y_pred_train = model(X_train)
        y_pred_validate = model(X_validate)

        loss_train = loss_fn(y_pred_train, y_train)
        loss_validate = loss_fn(y_pred_validate, y_validate)

        training_losses.append(loss_train.item())
        validation_losses.append(loss_validate.item())

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        if epoch % 20 == 0:

            print(f"Epoch {epoch}, Training Loss: {loss_train.item()}\n")
            print(f"Epoch {epoch}, Validation Loss: {loss_validate.item()}\n")

    return training_losses, validation_losses

def plot_losses(epochs, training_losses, validation_losses):

    plt.plot(range(epochs), training_losses, label = "Training Losses", color = "blue")
    plt.plot(range(epochs), validation_losses, label = "Validation Losses", color = "orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Time")
    plt.grid(True)
    plt.legend()
    plt.show()

chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")
df = chicago_taxi_dataset[["TRIP_MILES", "TRIP_SECONDS", "FARE", "COMPANY", "PAYMENT_TYPE", "TIP_RATE"]]

y_label = "FARE"
x_labels = get_xlabels(df, y_label, 0.7)

train, validate, test = np.split(df.sample(frac = 1), [int(0.6 * len(df.index)), int(0.8 * len(df.index))])

X_train, y_train = get_xy(train, y_label, x_labels)
X_validate, y_validate = get_xy(validate, y_label, x_labels)
X_test, y_test = get_xy(test, y_label, x_labels)

scaler = StandardScaler()

X_train = torch.tensor(scaler.fit_transform(X_train), dtype = torch.float32)
y_train = torch.tensor(y_train, dtype = torch.float32)

X_validate = torch.tensor(scaler.transform(X_validate), dtype = torch.float32)
y_validate = torch.tensor(y_validate, dtype = torch.float32)

X_test = torch.tensor(scaler.transform(X_test), dtype = torch.float32)
y_test = torch.tensor(y_test, dtype = torch.float32)

model = LinearRegressionModel(len(x_labels))
loss_fn = nn.MSELoss()
lr = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr, momentum = 0.9)
epochs = 100

training_losses, validation_losses = train_loop(model, X_train, y_train, X_validate, y_validate, loss_fn, optimizer, epochs)

plot_losses(epochs, training_losses, validation_losses)

new_lossfn_of_interest = nn.L1Loss()
y_pred_test = model(X_test)
loss_test = new_lossfn_of_interest(y_pred_test, y_test)
print(f"on average there is a deviation of ${loss_test.item()} from the true fare amount")