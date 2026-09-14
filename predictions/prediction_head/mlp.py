import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class TorchMLPRegressor:
    """Small PyTorch MLP regressor used as the camera-ready prediction head."""

    def __init__(
        self,
        hidden_layer_sizes=(16, 8),
        batch_size=256,
        learning_rate=1e-3,
        weight_decay=1e-3,
        max_epochs=300,
        validation_fraction=0.1,
        patience=15,
        random_state=42,
        verbose=False,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose

        self.imputer = SimpleImputer(strategy="constant", fill_value=0.0)
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def _build_model(self, input_dim):
        layers = []
        previous_dim = input_dim

        for hidden_dim in self.hidden_layer_sizes:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        X_np = self.imputer.fit_transform(X)
        X_np = self.scaler.fit_transform(X_np).astype(np.float32)
        y_np = np.asarray(y, dtype=np.float32)

        X_train, X_val, y_train, y_val = train_test_split(
            X_np,
            y_np,
            test_size=self.validation_fraction,
            random_state=self.random_state,
        )

        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.model = self._build_model(input_dim=X_np.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.MSELoss()

        X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=self.device)

        best_val_loss = float("inf")
        best_state_dict = None
        epochs_without_improvement = 0

        for epoch in range(self.max_epochs):
            self.model.train()
            train_losses = []

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                prediction = self.model(xb).squeeze(-1)
                loss = loss_fn(prediction, yb)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            self.model.eval()
            with torch.no_grad():
                val_prediction = self.model(X_val_tensor).squeeze(-1)
                val_loss = loss_fn(val_prediction, y_val_tensor).item()

            if self.verbose:
                mean_train_loss = float(np.mean(train_losses))
                print(
                    f"Epoch {epoch + 1:03d} | "
                    f"train_loss={mean_train_loss:.6f} | "
                    f"val_loss={val_loss:.6f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                break

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet")

        X_np = self.imputer.transform(X)
        X_np = self.scaler.transform(X_np).astype(np.float32)
        X_tensor = torch.tensor(X_np, dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(X_tensor).squeeze(-1)

        return prediction.cpu().numpy()


def build_prediction_head(random_state=42):
    """Build the PyTorch MLP prediction head used in the final experiments."""
    return TorchMLPRegressor(
        hidden_layer_sizes=(16, 8),
        batch_size=256,
        learning_rate=1e-3,
        weight_decay=1e-3,
        max_epochs=300,
        validation_fraction=0.1,
        patience=15,
        random_state=random_state,
        verbose=False,
    )
