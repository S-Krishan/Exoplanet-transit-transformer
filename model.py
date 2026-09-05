"""
Model definitions and training/evaluation utilities for the light-curve transformer.
"""

import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split


class LightCurveDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# Remember shape [a,b] means b columns and a rows
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2001):
        super().__init__()
        # Create a 2-d tensor filled with zeros with dimensions [rows,columns] -> [max_len,d_model]
        pe = torch.zeros(max_len, d_model)
        # Here we create a 1-d tensor of the numbers 0-max_len to represent the positions in the PE formula.
        # We do .unsqueeze(1) to add a extra dimension at position 1 so now theres 2001 rows with 1 column each to allow broadcasting with div_term
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Here we create an 1-d tensor which creates d_model/2 terms which consist of the denominator in the PE formula. The initial torch.arange() array is the i terms which are then manipulated to become the complete denominator.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        # Here we utilise broadcasting where position is copied across all columns and div_term is copied across all rows.
        # Now each row in the 2-d tensor represents a position whilst each column is the position multiplied by div term.
        # There are d_model/2 columns because we will represent only half the overall positions with this 2-d tensor (one half used for even positions and one half used for odd positions)
        position_over_div_term = position * div_term

        # Here we apply sine to even columns in the 2-d tensor and apply cos to odd columns in the 2-d tensor to complete the PE matrix
        pe[:, 0::2] = torch.sin(position_over_div_term)
        pe[:, 1::2] = torch.cos(position_over_div_term)

        # Add batch dimension to allow the PE tensor to broadcast with the input tensor
        pe = pe.unsqueeze(0)

        # register the pe tensor as a buffer so that it isn't treated as a parameter but it is stored as part of the model
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Here we slice the rows of the positional encoding tensor to x.size(1) (the sequence length).
        # We add the positional information to the x input using broadcasting to copy the PE tensor for each batch
        x = x + self.pe[:, :x.size(1)]
        return x


# nn.module makes it a pytorch neural network subclass which means it inherits __call__ so automatically runs forward
class LightCurveTransformer(nn.Module):
    def __init__(self, d_model=128, no_heads=4):
        # Make a class into a PyTorch module by inheriting attributes .parameters .modules etc.
        super().__init__()
        # Project 1-dimensional input into d_model-dimensional space. The dimension is referring to the inner-most dimension of the input as this is what is being transformed.
        self.embedding = nn.Linear(1, d_model)
        # Add positional encoding so model knows information on the location of each token
        self.positional_encoding = PositionalEncoding(d_model)
        # Define one encoding layer in the transformer with {nhead} attention heads
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=no_heads, batch_first=True)
        # Here we stack {num_layers} blocks to create the full encoder
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # This is the final classification layer which takes the vector and aims to output 0 for no transit or 1 for transit
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # Unsqueeze adds a extra dimension so that each flux value is a vector of dimension 1 because embedding expected input of shape [...,1]
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        # Here we average all 200 flux vectors to represent all of them in a single vector per batch which we can use for classification
        x = x.mean(dim=1)
        out = self.fc(x)
        return out


def train_validation_split(dataset):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    return train_loader, val_loader


def split_TESS_dataset(metadata_csv, seed=42):
    df = pd.read_csv(metadata_csv)
    # Create a table of grouped tic_ids and select just the label column(just take the row associated with the first label in each group)
    tic_id_labels = df.groupby("tic_id")["label"].first()
    unique_tics = tic_id_labels.index.values
    labels = tic_id_labels.values
    train_tics, temp_tics, train_labels, temp_labels = train_test_split(
        unique_tics, labels, test_size=0.3, stratify=labels, random_state=seed
    )
    val_tics, test_tics, _, _ = train_test_split(
        temp_tics, temp_labels, test_size=0.5, stratify=temp_labels, random_state=seed
    )
    train_df = df[df["tic_id"].isin(train_tics)]
    val_df = df[df["tic_id"].isin(val_tics)]
    test_df = df[df["tic_id"].isin(test_tics)]
    train_df.to_csv("TESS-train.csv", index=False)
    val_df.to_csv("TESS-val.csv", index=False)
    test_df.to_csv("TESS-test.csv", index=False)
    print(f"Train: {train_df['tic_id'].nunique()} stars, {len(train_df)} lightcurves")
    print(f"Val: {val_df['tic_id'].nunique()} stars, {len(val_df)} lightcurves")
    print(f"Test: {test_df['tic_id'].nunique()} stars, {len(test_df)} lightcurves")
    return train_df, val_df, test_df


def train(model, num_epochs, train_loader, val_loader, device):
    # The loss function is Binary Cross-Entropy Loss which measures how close a model's predicted probability is to a binary label (transit/no transit)
    criterion = nn.BCEWithLogitsLoss()
    # The optimiser is Adam which adjusts the step size for each individual weight for faster and more stable training
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="max", factor=0.5, patience=2)
    val_acc = evaluate_performance(model, val_loader, device)
    for epoch in range(0, num_epochs):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimiser.zero_grad()

            y_pred = model(x_batch)
            # We squeeze the second dimension because the model outputs a prediction as a 1-d tensor when we want a scalar value to prevent shape mismatch
            loss = criterion(y_pred.squeeze(1), y_batch.float())
            loss.backward()
            optimiser.step()
            running_loss += loss.item()
        average_loss = running_loss / len(train_loader)
        val_acc = evaluate_performance(model, val_loader, device)
        scheduler.step(val_acc)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}: {average_loss:.4f}, LR: {current_lr:.6f}")
    print("Training finished. Saving model and evaluating results...")
    torch.save(model.state_dict(), "lightcurve_transformer_5.pth")


def load_model():
    model = LightCurveTransformer()
    model.load_state_dict(torch.load("lightcurve_transformer_5.pth"))
    model.eval()
    return model


def evaluate_performance(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            # Get the logit output for each lightcurve
            y_pred = model(x_batch)
            # Convert the logit outputs into probabilities between 0 and 1 using sigmoid
            y_prob = torch.sigmoid(y_pred.squeeze(1))
            # This allows us to make a comparison with 0.5 to show if it thinks there is a planet or not (big number squashes into >0.5)
            # .long converts Boolean into integers 1 or 0
            preds = (y_prob > 0.5).long()
            # We compare this batch to the actual values to and add correct values to tally
            correct += (preds == y_batch.long()).sum().item()
            # Add how many samples were in this batch to total
            total += y_batch.size(0)
    # accuracy is samples correct/total samples
    acc = correct / total
    print(f"Validation Accuracy: {acc:.4f}")
    return acc
