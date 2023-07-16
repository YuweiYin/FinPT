# -*- coding: utf-8 -*-

from abc import ABC
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from model.basemodel import BaseModel
from utils.io_utils import get_output_path


class BaseModelTorch(BaseModel, ABC):

    def __init__(self, params=None, args=None):
        super().__init__(params, args)
        self.device = self.get_device()
        self.gpus = args.cuda if args.cuda != "cpu" and torch.cuda.is_available() and args.data_parallel else None

    def to_device(self):
        if self.args.data_parallel:
            self.model = nn.DataParallel(self.model, device_ids=self.args.gpu_ids)

        print("On Device:", self.device)
        self.model.to(self.device)

    def get_device(self):
        if hasattr(self.args, "device"):
            return torch.device(self.args.device)

        if self.args.cuda != "cpu" and torch.cuda.is_available():
            if self.args.data_parallel:
                device = "cuda"  # + "".join(str(i) + "," for i in self.args.gpu_ids)[:-1]
            else:
                device = "cuda"
        else:
            device = "cpu"

        return torch.device(device)

    def fit(self, X, y, X_val=None, y_val=None, optimizer=None, criterion=None):
        if optimizer is None:
            optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr)

        X = torch.tensor(X).float().to(self.device)
        X_val = torch.tensor(X_val).float().to(self.device)

        y = torch.tensor(y).to(self.device)
        y_val = torch.tensor(y_val).to(self.device)

        # if self.args.objective == "regression":
        #     loss_func = nn.MSELoss()
        #     y = y.float()
        #     y_val = y_val.float()
        # elif self.args.objective == "classification":
        #     loss_func = nn.CrossEntropyLoss()
        # else:
        #     loss_func = nn.BCEWithLogitsLoss()
        #     y = y.float()
        #     y_val = y_val.float()

        if criterion is None:
            if self.args.objective == "regression":
                criterion = nn.MSELoss()
            else:
                criterion = nn.CrossEntropyLoss()

        if self.args.objective == "regression":
            y = y.float()
            y_val = y_val.float()
        else:
            y = y.long()
            y_val = y_val.long()

        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.args.bsz, shuffle=True, num_workers=4)

        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.args.bsz, shuffle=True)

        min_val_loss = float("inf")
        min_val_loss_idx = 0

        loss_history = []
        val_loss_history = []

        for epoch in range(self.args.epoch):
            for i, (batch_X, batch_y) in enumerate(train_loader):
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                out = self.model(batch_X)

                if self.args.objective == "regression" or self.args.objective == "binary":
                    out = out.squeeze()

                loss = criterion(out, batch_y)
                loss_history.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Early Stopping
            val_loss = 0.0
            val_dim = 0
            for val_i, (batch_val_X, batch_val_y) in enumerate(val_loader):
                batch_val_X, batch_val_y = batch_val_X.to(self.device), batch_val_y.to(self.device)
                out = self.model(batch_val_X)

                if self.args.objective == "regression" or self.args.objective == "binary":
                    out = out.squeeze()

                val_loss += criterion(out, batch_val_y)
                val_dim += 1

            val_loss /= val_dim
            val_loss_history.append(val_loss.item())

            print("Epoch %d, Val Loss: %.5f" % (epoch, val_loss))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_loss_idx = epoch

                # Save the currently best model
                self.save_model(filename_extension="best", directory="tmp")

            # if min_val_loss_idx + self.args.early_stopping_rounds < epoch:
            #     print("Validation loss has not improved for %d steps!" % self.args.early_stopping_rounds)
            #     print("Early stopping applies.")
            #     break

        # Load best model
        self.load_model(filename_extension="best", directory="tmp")
        return loss_history, val_loss_history

    def evaluate(self, X, y):
        preds = self.predict(X)
        # preds = np.argmax(preds, axis=1)
        eval_result = dict({})
        eval_result["accuracy"] = accuracy_score(y, preds)
        eval_result["f1"] = f1_score(y, preds)
        return eval_result

    def predict(self, X):
        if self.args.objective == "regression":
            self.predictions = self.predict_helper(X)
        else:
            self.predict_proba(X)
            self.predictions = np.argmax(self.prediction_probabilities, axis=1)

        return self.predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_helper(X)

        # If binary task returns only probability for the true class, adapt it to return (N x 2)
        if probas.shape[1] == 1:
            probas = np.concatenate((1 - probas, probas), 1)

        self.prediction_probabilities = probas
        return self.prediction_probabilities

    def predict_helper(self, X):
        self.model.eval()

        X = torch.tensor(X).float()
        test_dataset = TensorDataset(X)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.args.bsz, shuffle=False, num_workers=2)
        predictions = []
        with torch.no_grad():
            for batch_X in test_loader:
                preds = self.model(batch_X[0].to(self.device))

                if self.args.objective == "binary":
                    preds = torch.sigmoid(preds)

                predictions.append(preds.detach().cpu().numpy())
        return np.concatenate(predictions)

    def save_model(self, filename_extension="", directory="models"):
        filename = get_output_path(self.args, directory=directory, filename="m", extension=filename_extension,
                                   file_type="pt")
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename_extension="", directory="models"):
        filename = get_output_path(self.args, directory=directory, filename="m", extension=filename_extension,
                                   file_type="pt")
        state_dict = torch.load(filename)
        self.model.load_state_dict(state_dict)

    def get_model_size(self):
        model_size = sum(t.numel() for t in self.model.parameters() if t.requires_grad)
        return model_size

    @classmethod
    def define_trial_parameters(cls, trial, args):
        raise NotImplementedError("This method has to be implemented by the sub class")
