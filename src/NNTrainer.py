import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import StepLR

import numpy as np

import Splitter

import os
import time
import random
import math


class NNTrainer:

    def __init__(self, token, target, vector_model, embedding_size=100, scope_before=5, scope_after=5):
        self.vector_model = vector_model
        self.embedding_size = embedding_size
        self.scope_before = scope_before
        self.scope_after = scope_after
        self.dataset = WindowDataset(token, target, self.vector_model, self.embedding_size,
                                     scope_before=self.scope_before, scope_after=self.scope_after)
        self.train_dataset = self.dataset
        self.test_dataset = self.dataset
        self.valid_dataset = self.dataset

    def add_train_samples(self, token, target):
        new_dataset = WindowDataset(token, target, self.vector_model, self.embedding_size,
                                    scope_before=self.scope_before, scope_after=self.scope_after)
        self.dataset = self.dataset + new_dataset
        self.train_dataset = self.dataset
        self.test_dataset = self.dataset
        self.valid_dataset = self.dataset

    def train(self, param):
        # Create the NN
        print("Training the RNN Modell: ", param)
        if param["save"]:
            model_path = os.path.join("Models", (param["log_file"][:-3] + "model"))
        start_time = time.time()
        best_performance = {'f1': 0.0, 'recall': 0.0, 'precision': 0.0}
        lr = param["lr"]

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        cpu = torch.device('cpu')
        if param["cuda"] and not torch.cuda.is_available():
            print("Using CPU")
            param["cuda"] = False

        class_weights = torch.tensor([3], dtype=torch.float32)
        if param["cuda"]:
            class_weights = class_weights.cuda()

        message = ""
        for k, v in param.items():
            message += "{} {}\n".format(k, v)
        print(message)
        if param["log_file"]:
            with open(os.path.join("Logs", param["log_file"]), "a") as f:
                f.write(message)
                f.flush()

        #np.random.seed(param["seed"])
        #torch.manual_seed(param["seed"])
        model = GruWindow2(d_in=param["input_dim"], d_h=param["hidden_dim"], num_layers=param["layers"],
                           bidirectional=param["bidirectional"],
                           scope_before=param["scope_before"], scope_after=param["scope_after"])

        # Initialize the datasplits
        valid_length = math.ceil(len(self.dataset) * param["valid_percentage"])
        test_length = math.ceil(len(self.dataset) * param["test_percentage"])
        train_length = len(self.dataset) - valid_length - test_length
        subsetlist = random_split(self.dataset, [train_length, valid_length, test_length])
        self.train_dataset = subsetlist[0]
        self.valid_dataset = subsetlist[1]
        self.test_dataset = subsetlist[2]
        print("Train dataset:", len(self.train_dataset))
        print("Valid dataset:", len(self.valid_dataset))
        print("Test  dataset:", len(self.test_dataset))
        # Initialize the dataloader
        dataloader = DataLoader(self.train_dataset, batch_size=512, shuffle=True)
        validloader = DataLoader(self.valid_dataset, batch_size=128)
        testloader = DataLoader(self.test_dataset, batch_size=128)
        # Initialize the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if param["lr_decay"]:
            scheduler = StepLR(optimizer, step_size=5, gamma=0.7)
        #loss = nn.BCELoss(weight=class_weights)
        loss = nn.BCELoss()

        if param["cuda"]:
            model.cuda()

        print("Network")
        print(model)
        with open(os.path.join("Logs", param["log_file"]), "a") as f:
            f.write(str(model))
            f.flush()

        # Run over epochs
        for e in range(param["epochs"]):

            if param["lr_decay"] and not (e + 1) > 50:
                scheduler.step()

            epoch_time = time.time()

            agg_cost = torch.tensor([0.])
            if param["cuda"]:
                agg_cost.cuda()
            num_batches = 0
            model.train()
            # Load the data
            # Use dataloader that randomizes the order of the data subsets.
            for i, (train_token, train_target) in enumerate(dataloader):
                if param["cuda"]:
                    train_token = train_token.cuda()
                    train_target = train_target.cuda()
                optimizer.zero_grad()

                # pass of data through network
                output_target = model.forward(train_token)
                # Calculate the loss
                cost = loss(output_target, train_target)
                agg_cost += cost

                cost.backward()
                optimizer.step()

                num_batches += 1

            # evaluation
            model.eval()
            true_pos = 0
            false_pos = 0
            true_neg = 0
            false_neg = 0
            val_cost = np.array([0.0])
            for test_token, test_label in validloader:
                if param["cuda"]:
                    test_token = test_token.cuda()
                    test_label = test_label.cuda()
                output = model.forward(test_token)
                val_cost += loss(output, test_label).to(cpu).detach().numpy()
                output = output.to(cpu).detach().numpy()
                preds = get_prediction(output)
                true_label = test_label.to(cpu).detach().numpy()

                true_pos += np.sum((true_label == preds) & (true_label == 1))
                false_pos += np.sum((true_label != preds) & (true_label == 0))
                true_neg += np.sum((true_label == preds) & (true_label == 0))
                false_neg += np.sum((true_label != preds) & (true_label == 1))
            total = true_pos + true_neg + false_pos + false_neg
            pos = true_pos + false_neg
            neg = true_neg + false_pos
            sens = true_pos / (true_pos + false_neg) if true_pos + false_neg > 0 else 0
            spec = true_neg / (false_pos + true_neg) if false_pos + true_neg > 0 else 0
            prec = true_pos / (true_pos + false_pos) if true_pos + false_pos > 0 else 0
            npv = true_neg / (true_neg + false_neg) if true_neg + false_neg > 0 else 0
            fpr = false_pos / (false_pos + true_neg) if false_pos + true_neg > 0 else 1
            fdr = false_pos / (false_pos + true_pos) if false_pos + true_pos > 0 else 1
            fnr = false_neg / (false_neg + true_pos) if false_neg + true_pos > 0 else 1
            f1 = 2 * true_pos / (
                        2 * true_pos + false_pos + false_neg) if 2 * true_pos + false_pos + false_neg > 0 else 0
            acc = (true_pos + true_neg) / total if total > 0 else 0
            bacc = (true_pos / pos + true_neg / neg) / 2 if pos > 0 and neg > 0 else 0
            evaluation = {"recall": sens,
                          "specificity": spec,
                          "precision": prec,
                          "npv": npv,
                          "fpr": fpr,
                          "fdr": fdr,
                          "fnr": fnr,
                          "acc": acc,
                          "bacc": bacc,
                          "f1": f1}
            if evaluation["f1"] > best_performance["f1"]:
                if param["save"]:
                    torch.save(model, model_path)
                print("Wrote new best model")
                best_performance["f1"] = evaluation["f1"]
            if evaluation["recall"] > best_performance["recall"]:
                best_performance["recall"] = evaluation["recall"]
            if evaluation["precision"] > best_performance["precision"]:
                best_performance["precision"] = evaluation["precision"]
            message = ("*** Epoch: {} ***\n" +
                       "* F1   {:.5f} [{}]\n"
                       "* REC  {:.5f} [{}]\n"
                       "* PRE  {:.5f} [{}]\n"
                       "* SPE  {:.5f}\n"
                       "* ACC  {:.5f}\n"
                       "* NPV  {:.5f}\n"
                       "* FPR  {:.5f}\n"
                       "* FNR  {:.5f}\n"
                       "* BACC {:.5f}\n").format(
                e + 1,
                evaluation["f1"], best_performance["f1"],
                evaluation["recall"], best_performance["recall"],
                evaluation["precision"], best_performance["precision"],
                evaluation["specificity"],
                evaluation["acc"],
                evaluation["npv"],
                evaluation["fpr"],
                evaluation["fnr"],
                evaluation["bacc"]
            )
            print(message)
            if param["log_file"]:
                with open(os.path.join("Logs", param["log_file"]), 'a') as f:
                    f.write(message)
                    f.flush()
            train_cost = agg_cost.to(cpu).detach().numpy()
            message = "Train Cost: " + str(train_cost[0]) + "\n"
            print(message)
            if param["log_file"]:
                with open(os.path.join("Logs", param["log_file"]), 'a') as f:
                    f.write(message)
                    f.flush()
            message = "Valid Cost: " + str(val_cost[0]) + "\n"
            print(message)
            if param["log_file"]:
                with open(os.path.join("Logs", param["log_file"]), 'a') as f:
                    f.write(message)
                    f.flush()
            epoch_message = " Time: {} min".format((time.time() - epoch_time) / 60)
            print(epoch_message)
        message = ("*** Finished ***\n"
                   " Result: {}\n"
                   " Time: {} min\n").format(
            best_performance,
            (time.time() - start_time) / 60
        )
        print(message)
        if param["log_file"]:
            with open(os.path.join("Logs", param["log_file"]), 'a') as f:
                f.write(message)
                f.flush()
        # evaluation
        if param["save"]:
            model = torch.load(model_path)
        model.eval()
        true_pos = 0
        false_pos = 0
        true_neg = 0
        false_neg = 0
        for test_token, test_label in testloader:
            if param["cuda"]:
                test_token = test_token.cuda()
                test_label = test_label.cuda()
            output = model.forward(test_token)
            output = output.to(cpu).detach().numpy()
            preds = get_prediction(output)
            true_label = test_label.to(cpu).detach().numpy()

            true_pos += np.sum((true_label == preds) & (true_label == 1))
            false_pos += np.sum((true_label != preds) & (true_label == 0))
            true_neg += np.sum((true_label == preds) & (true_label == 0))
            false_neg += np.sum((true_label != preds) & (true_label == 1))
        total = true_pos + true_neg + false_pos + false_neg
        pos = true_pos + false_neg
        neg = true_neg + false_pos
        sens = true_pos / (true_pos + false_neg) if true_pos + false_neg > 0 else 0
        prec = true_pos / (true_pos + false_pos) if true_pos + false_pos > 0 else 0
        f1 = 2 * true_pos / (
                2 * true_pos + false_pos + false_neg) if 2 * true_pos + false_pos + false_neg > 0 else 0
        acc = (true_pos + true_neg) / total if total > 0 else 0
        bacc = (true_pos / pos + true_neg / neg) / 2 if pos > 0 and neg > 0 else 0
        evaluation = {"recall": sens,
                      "precision": prec,
                      "acc": acc,
                      "bacc": bacc,
                      "f1": f1}
        message = ("*** Test Perf. ***\n" +
                   "* F1   {:.5f}\n"
                   "* REC  {:.5f}\n"
                   "* PRE  {:.5f}\n"
                   "* ACC  {:.5f}\n"
                   "* BACC {:.5f}\n").format(
            evaluation["f1"],
            evaluation["recall"],
            evaluation["precision"],
            evaluation["acc"],
            evaluation["bacc"]
        )
        print(message)
        if param["log_file"]:
            with open(os.path.join("Logs", param["log_file"]), 'a') as f:
                f.write(message)
                f.flush()

        return best_performance

    def create_train_test_split(self, token, target):
        """
        Splits the document up into a train and a test part.
        :param token: The token
        :param target: The corresponding target for every token
        :return: dictionary with X_train, Y_train, X_text, Y_test as keys
        """
        assert len(token) == len(target)
        test_percentage = 0.3
        start_index = random.randint(1, len(token) - 1)
        size = math.ceil(test_percentage * len(token))
        end_index = (start_index + size) % len(token)
        dic = {}
        if end_index < start_index:
            dic["X_test"] = token[start_index:] + token[:end_index]
            dic["X_train"] = token[end_index:start_index]
            dic["Y_test"] = target[start_index:] + target[:end_index]
            dic["Y_train"] = target[end_index:start_index]
        else:
            dic["X_test"] = token[start_index:end_index]
            dic["X_train"] = token[end_index:] + token[:start_index]
            dic["Y_test"] = target[start_index:end_index]
            dic["Y_train"] = target[end_index:] + target[:start_index]
        return dic

    def get_data_split(self, dataset, val_percentage, test_percentage):
        """
        Generates the train, validation, test split for the dataset.
        :param dataset:
        :param val_percentage:
        :param test_percentage:
        :return:
        """


class RNNNN(nn.Module):

    def __init__(self, d_in=100, d_h=20, num_layers=1, bidirectional=True):
        super(RNNNN, self).__init__()
        self.rnn = nn.RNN(input_size=d_in, hidden_size=d_h, num_layers=num_layers, bidirectional=bidirectional)
        self.linear = nn.Linear(in_features=d_h*(2 if bidirectional else 1), out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, tok):
        x, h_n = self.rnn(tok)
        x = self.linear(x)
        x = self.sig(x)
        return x


class LSTMNN(nn.Module):

    def __init__(self, d_in=100, d_h=20, num_layers=1, bidirectional=True):
        super(LSTMNN, self).__init__()
        self.rnn = nn.LSTM(input_size=d_in, hidden_size=d_h, num_layers=num_layers, bidirectional=bidirectional)
        self.linear = nn.Linear(in_features=d_h * (2 if bidirectional else 1), out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, tok):
        x, h_n = self.rnn(tok)
        x = self.linear(x)
        x = self.sig(x)
        return x


class GruNN(nn.Module):

    def __init__(self, d_in=100, d_h=20, num_layers=1, bidirectional=True):
        super(GruNN, self).__init__()
        self.rnn = nn.GRU(input_size=d_in, hidden_size=d_h, num_layers=num_layers, bidirectional=bidirectional)
        self.linear = nn.Linear(in_features=d_h * (2 if bidirectional else 1), out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, tok):
        x, h_n = self.rnn(tok)
        x = self.linear(x)
        x = self.sig(x)
        return x


class RNNWindow(nn.Module):

    def __init__(self, d_in=100, d_h=20, num_layers=1, bidirectional=True, scope_before=3, scope_after=3):
        super(RNNWindow, self).__init__()
        self.scope_before = scope_before
        self.scope_after = scope_after
        self.in_features = d_h*(2 if bidirectional else 1)*(scope_before+scope_after+1)
        self.rnn = nn.RNN(input_size=d_in, hidden_size=d_h, num_layers=num_layers, bidirectional=bidirectional)
        self.linear = nn.Linear(in_features=self.in_features, out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, tok):
        x, _ = self.rnn(tok)
        x = x.view(-1, self.in_features)
        x = self.linear(x)
        x = self.sig(x)
        return x


class RNNWindow2(nn.Module):

    def __init__(self, d_in=100, d_h=20, num_layers=1, bidirectional=True, scope_before=3, scope_after=3):
        super(RNNWindow2, self).__init__()
        self.scope_before = scope_before
        self.scope_after = scope_after
        self.in_features = d_h*(2 if bidirectional else 1)*(scope_before+scope_after+1)
        self.rnn = nn.RNN(input_size=d_in, hidden_size=d_h, num_layers=num_layers, bidirectional=bidirectional)
        self.linear = nn.Linear(in_features=self.in_features, out_features=d_h)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=d_h, out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, tok):
        x, _ = self.rnn(tok)
        x = x.view(-1, self.in_features)
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sig(x)
        return x


class LSTMWindow(nn.Module):

    def __init__(self, d_in=100, d_h=20, num_layers=1, bidirectional=True, scope_before=3, scope_after=3):
        super(LSTMWindow, self).__init__()
        self.scope_before = scope_before
        self.scope_after = scope_after
        self.in_features = d_h*(2 if bidirectional else 1)*(scope_before+scope_after+1)
        self.rnn = nn.LSTM(input_size=d_in, hidden_size=d_h, num_layers=num_layers, bidirectional=bidirectional)
        self.linear = nn.Linear(in_features=self.in_features, out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, tok):
        x, _ = self.rnn(tok)
        x = x.view(-1, self.in_features)
        x = self.linear(x)
        x = self.sig(x)
        return x


class LSTMWindow2(nn.Module):

    def __init__(self, d_in=100, d_h=20, num_layers=1, bidirectional=True, scope_before=3, scope_after=3):
        super(LSTMWindow2, self).__init__()
        self.scope_before = scope_before
        self.scope_after = scope_after
        self.in_features = d_h*(2 if bidirectional else 1)*(scope_before+scope_after+1)
        self.rnn = nn.LSTM(input_size=d_in, hidden_size=d_h, num_layers=num_layers, bidirectional=bidirectional)
        self.linear = nn.Linear(in_features=self.in_features, out_features=d_h)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=d_h, out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, tok):
        x, _ = self.rnn(tok)
        x = x.view(-1, self.in_features)
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sig(x)
        return x


class LSTMWindow4(nn.Module):

    def __init__(self, d_in=100, d_h=20, lin_d=[32, 16], num_layers=1, bidirectional=True, scope_before=3, scope_after=3):
        super(LSTMWindow4, self).__init__()
        self.scope_before = scope_before
        self.scope_after = scope_after
        self.in_features = d_h*(2 if bidirectional else 1)*(scope_before+scope_after+1)
        self.rnn = nn.LSTM(input_size=d_in, hidden_size=d_h, num_layers=num_layers, bidirectional=bidirectional)
        self.linear = nn.Linear(in_features=self.in_features, out_features=d_h)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=d_h, out_features=lin_d[0])
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(in_features=lin_d[0], out_features=lin_d[1])
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(in_features=lin_d[1], out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, tok):
        x, _ = self.rnn(tok)
        x = x.view(-1, self.in_features)
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        x = self.sig(x)
        return x


class GruWindow(nn.Module):

    def __init__(self, d_in=100, d_h=20, num_layers=1, bidirectional=True, scope_before=3, scope_after=3):
        super(GruWindow, self).__init__()
        self.scope_before = scope_before
        self.scope_after = scope_after
        self.in_features = d_h*(2 if bidirectional else 1)*(scope_before+scope_after+1)
        self.rnn = nn.GRU(input_size=d_in, hidden_size=d_h, num_layers=num_layers, bidirectional=bidirectional)
        self.linear = nn.Linear(in_features=self.in_features, out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, tok):
        x, _ = self.rnn(tok)
        x = x.view(-1, self.in_features)
        x = self.linear(x)
        x = self.sig(x)
        return x


class GruWindow2(nn.Module):

    def __init__(self, d_in=100, d_h=20, num_layers=1, bidirectional=True, scope_before=3, scope_after=3):
        super(GruWindow2, self).__init__()
        self.scope_before = scope_before
        self.scope_after = scope_after
        self.in_features = d_h*(2 if bidirectional else 1)*(scope_before+scope_after+1)
        self.rnn = nn.GRU(input_size=d_in, hidden_size=d_h, num_layers=num_layers, bidirectional=bidirectional)
        self.linear = nn.Linear(in_features=self.in_features, out_features=d_h)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=d_h, out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, tok):
        x, _ = self.rnn(tok)
        x = x.view(-1, self.in_features)
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sig(x)
        return x


class GruWindow4(nn.Module):

    def __init__(self, d_in=100, d_h=20, lin_d=[32, 16], num_layers=1, bidirectional=True, scope_before=3, scope_after=3):
        super(GruWindow4, self).__init__()
        self.scope_before = scope_before
        self.scope_after = scope_after
        self.in_features = d_h*(2 if bidirectional else 1)*(scope_before+scope_after+1)
        self.rnn = nn.GRU(input_size=d_in, hidden_size=d_h, num_layers=num_layers, bidirectional=bidirectional)
        self.linear = nn.Linear(in_features=self.in_features, out_features=d_h)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=d_h, out_features=lin_d[0])
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(in_features=lin_d[0], out_features=lin_d[1])
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(in_features=lin_d[1], out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, tok):
        x, _ = self.rnn(tok)
        x = x.view(-1, self.in_features)
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        x = self.sig(x)
        return x


class RNN2NN(nn.Module):

    def __init__(self, d_in=100, d_h=20, num_layers=1, bidirectional=True, scope_before=3, scope_after=3):
        super(RNN2NN, self).__init__()
        self.scope_before = scope_before
        self.scope_after = scope_after
        self.in_features = d_h * (2 if bidirectional else 1) * (scope_before + scope_after + 1)
        self.rnn = nn.RNN(input_size=d_in, hidden_size=d_h, num_layers=num_layers, bidirectional=bidirectional)
        self.linear = nn.Linear(in_features=self.in_features, out_features=d_h)
        self.linear2 = nn.Linear(in_features=d_h, out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, tok):
        x, h_n = self.rnn(tok)
        x = x.view(-1, self.in_features)
        x = self.linear(x)
        x = self.linear2(x)
        x = self.sig(x)
        return x


class LSTM2NN(nn.Module):

    def __init__(self, d_in=100, d_h=20, num_layers=1, bidirectional=True, scope_before=3, scope_after=3):
        super(LSTM2NN, self).__init__()
        self.scope_before = scope_before
        self.scope_after = scope_after
        self.in_features = d_h * (2 if bidirectional else 1) * (scope_before + scope_after + 1)
        self.rnn = nn.LSTM(input_size=d_in, hidden_size=d_h, num_layers=num_layers, bidirectional=bidirectional)
        self.linear = nn.Linear(in_features=self.in_features, out_features=d_h)
        self.linear2 = nn.Linear(in_features=d_h, out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, tok):
        x, h_n = self.rnn(tok)
        x = x.view(-1, self.in_features)
        x = self.linear(x)
        x = self.linear2(x)
        x = self.sig(x)
        return x


class LSTMPool(nn.Module):

    def __init__(self, d_in=100, d_h=20, num_layers=1, bidirectional=True, scope_before=3, scope_after=3):
        super(LSTMPool, self).__init__()
        self.scope_before = scope_before
        self.scope_after = scope_after
        self.in_features = 4*16*(2 if bidirectional else 1)
        self.rnn = nn.LSTM(input_size=d_in, hidden_size=d_h, num_layers=num_layers, bidirectional=bidirectional)
        self.cnn = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(4, 3), stride=(2, 2), padding=(1, 1))
        self.cnn2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.linear = nn.Linear(in_features=self.in_features, out_features=d_h)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=d_h, out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, tok):
        x, h_n = self.rnn(tok)
        x = torch.unsqueeze(x, dim=1)
        x = self.cnn(x)
        x = self.pool(x)
        x = self.cnn2(x)
        x = self.pool2(x)
        x = x.view(-1, self.in_features)
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sig(x)
        return x


class GRUPool(nn.Module):

    def __init__(self, d_in=100, d_h=20, num_layers=1, bidirectional=True, scope_before=3, scope_after=3):
        super(GRUPool, self).__init__()
        self.scope_before = scope_before
        self.scope_after = scope_after
        self.in_features = 4*16*(2 if bidirectional else 1)
        self.rnn = nn.GRU(input_size=d_in, hidden_size=d_h, num_layers=num_layers, bidirectional=bidirectional)
        self.cnn = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(4, 3), stride=(2, 2), padding=(1, 1))
        self.cnn2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.linear = nn.Linear(in_features=self.in_features, out_features=d_h)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=d_h, out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, tok):
        x, h_n = self.rnn(tok)
        x = torch.unsqueeze(x, dim=1)
        x = self.cnn(x)
        x = self.pool(x)
        x = self.cnn2(x)
        x = self.pool2(x)
        x = x.view(-1, self.in_features)
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sig(x)
        return x

class GRU2NN(nn.Module):

    def __init__(self, d_in=100, d_h=20, num_layers=1, bidirectional=True, scope_before=3, scope_after=3):
        super(GRU2NN, self).__init__()
        self.scope_before = scope_before
        self.scope_after = scope_after
        self.in_features = d_h * (2 if bidirectional else 1) * (scope_before + scope_after + 1)
        self.rnn = nn.GRU(input_size=d_in, hidden_size=d_h, num_layers=num_layers, bidirectional=bidirectional)
        self.linear = nn.Linear(in_features=self.in_features, out_features=d_h)
        self.linear2 = nn.Linear(in_features=d_h, out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, tok):
        x, h_n = self.rnn(tok)
        x = x.view(-1, self.in_features)
        x = self.linear(x)
        x = self.linear2(x)
        x = self.sig(x)
        return x


class RNNCNN(nn.Module):

    def __init__(self, d_in=100, d_h=20, num_layers=1, bidirectional=True, scope_before=3, scope_after=3):
        super(RNNCNN, self).__init__()
        self.scope_before = scope_before
        self.scope_after = scope_after
        self.in_features = d_h*(2 if bidirectional else 1)*(scope_before+scope_after+1)
        self.rnn = nn.RNN(input_size=d_in, hidden_size=d_h, num_layers=num_layers, bidirectional=bidirectional)
        self.cnn = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.linear = nn.Linear(in_features=self.in_features, out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, tok):
        x, _ = self.rnn(tok)
        x = torch.unsqueeze(x, dim=1)
        x = self.cnn(x)
        x = x.view(-1, self.in_features)
        x = self.linear(x)
        x = self.sig(x)
        return x


class LSTMCNN(nn.Module):

    def __init__(self, d_in=100, d_h=20, num_layers=1, bidirectional=True, scope_before=3, scope_after=3):
        super(LSTMCNN, self).__init__()
        self.scope_before = scope_before
        self.scope_after = scope_after
        self.in_features = d_h*(2 if bidirectional else 1)*(scope_before+scope_after+1)
        self.rnn = nn.LSTM(input_size=d_in, hidden_size=d_h, num_layers=num_layers, bidirectional=bidirectional)
        self.cnn = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.linear = nn.Linear(in_features=self.in_features, out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, tok):
        x, _ = self.rnn(tok)
        x = torch.unsqueeze(x, dim=1)
        x = self.cnn(x)
        x = x.view(-1, self.in_features)
        x = self.linear(x)
        x = self.sig(x)
        return x


class GRUCNN(nn.Module):

    def __init__(self, d_in=100, d_h=20, num_layers=1, bidirectional=True, scope_before=3, scope_after=3):
        super(GRUCNN, self).__init__()
        self.scope_before = scope_before
        self.scope_after = scope_after
        self.in_features = d_h*(2 if bidirectional else 1)*(scope_before+scope_after+1)
        self.rnn = nn.GRU(input_size=d_in, hidden_size=d_h, num_layers=num_layers, bidirectional=bidirectional)
        self.cnn = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.linear = nn.Linear(in_features=self.in_features, out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, tok):
        x, _ = self.rnn(tok)
        x = torch.unsqueeze(x, dim=1)
        x = self.cnn(x)
        x = x.view(-1, self.in_features)
        x = self.linear(x)
        x = self.sig(x)
        return x


class SBDDataset(Dataset):

    def __init__(self, token, target, vector_model, embedding_size):
        """
        :param token: All the training tokens
        :param target: The corresponding targets based on the sbd annotations
        """
        sentences = Splitter.split_tokens(token, target, filter_newline=False)
        self.token = []
        self.target = []
        target_index = 0
        for i, sentence in enumerate(sentences):
            # The last token of the last sentence needs to be appended
            if i == len(sentences) - 1:
                end_index = target_index + len(sentence) + 1
            else:
                end_index = target_index + len(sentence)
            tok = torch.tensor([vector_model[tok] if tok in vector_model else np.zeros(embedding_size)
                                for tok in sentence], dtype=torch.float32)
            self.token.append(tok)
            tar = torch.tensor([1 if t else 0 for t in target[target_index:end_index]], dtype=torch.float32)
            # One more dimension is need for the loss calculation
            tar = torch.unsqueeze(tar, 1)
            self.target.append(tar)
            assert tar.shape[0] == tok.shape[0]
            target_index = end_index

    def __getitem__(self, idx):
        token = self.token[idx]
        target = self.target[idx]

        return token, target

    def __len__(self):
        return len(self.token)


class WindowDataset(Dataset):

    def __init__(self, token, target, vector_model, embedding_size, scope_before=3, scope_after=3):
        """
        Same as other dataset, __getitem__ outpus a window
        :param token: All the training tokens
        :param target: The corresponding targets based on the sbd annotations
        """
        self.token = []
        self.target = []
        self.embedding_size = embedding_size
        self.scope_before = scope_before
        self.scope_after = scope_after
        for i, tok in enumerate(token):
            # Input accumulation
            token = vector_model[tok] if tok in vector_model else np.zeros(embedding_size)
            token_tensor = torch.tensor(token, dtype=torch.float32)
            self.token.append(token_tensor)
            # Target accumulation
            tag = 1 if target[i] else 0
            target_tensor = torch.tensor(tag, dtype=torch.float32)
            # One more dimension is need for the loss calculation
            target_tensor = torch.unsqueeze(target_tensor, -1)
            self.target.append(target_tensor)

    def __getitem__(self, idx):
        scope_start = idx - self.scope_before
        # We need + 1 because otherwise the last element in the window would not be included
        # scope_end is used in the range function as the delimiter
        scope_end = idx + self.scope_after + 1
        token = list()
        for s in range(scope_start, 0):
            token.append(torch.tensor(np.zeros(self.embedding_size), dtype=torch.float32))
        scope_start = max(scope_start, 0)
        for s in range(scope_start, idx):
            token.append(self.token[s])
        token.append(self.token[idx])
        for e in range(idx+1, min(self.__len__(), scope_end)):
            token.append(self.token[e])
        for e in range(self.__len__(), scope_end):
            token.append(torch.tensor(np.zeros(self.embedding_size), dtype=torch.float32))
        token = [torch.unsqueeze(t, dim=0) for t in token]
        output_token = torch.cat(token, dim=0)
        target = self.target[idx]

        return output_token, target

    def __len__(self):
        return len(self.token)


class SentenceLoader:
    """Dataloader used to load multiple sentences at once. Output is the concatenation."""
    def __init__(self, data, num_sentences=3):
        total_batches = math.ceil(len(data)/num_sentences)
        self.sentences = []
        self.targets = []
        # Go over all full batches
        for i in range(total_batches-1):
            sentences = []
            labels = []
            for s in range(num_sentences):
                token, target = data[i*num_sentences+s]
                sentences.append(token)
                labels.append(target)
            sent_cat = torch.cat(sentences, dim=0)
            sent_cat = torch.unsqueeze(sent_cat, dim=0)
            label_cat = torch.cat(labels, dim=0)
            label_cat = torch.unsqueeze(label_cat, dim=0)
            self.sentences.append(sent_cat)
            self.targets.append(label_cat)
        # Now we have to append the last batch (which is not entirly filled)
        sentences = []
        labels = []
        for i in range(len(self.sentences)*num_sentences, len(data)):
            token, target = data[i]
            sentences.append(token)
            labels.append(target)
        sent_cat = torch.cat(sentences, dim=0)
        sent_cat = torch.unsqueeze(sent_cat, dim=0)
        label_cat = torch.cat(labels, dim=0)
        label_cat = torch.unsqueeze(label_cat, dim=0)
        self.sentences.append(sent_cat)
        self.targets.append(label_cat)

    def __iter__(self):
        """Yield the sentences and the labels"""
        for s, t in zip(self.sentences, self.targets):
            yield s, t

    def __len__(self):
        """Number of possible sentences batches"""
        return len(self.sentences)


def get_prediction(x):
    dim = x.shape
    x = x.reshape(-1)
    ret = np.array([1 if i > 0.5 else 0 for i in x])
    ret = ret.reshape(dim)
    return ret

