import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import StepLR

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

import numpy as np

import os
import time
import random
import math
import codecs
import json
import re
from collections import deque



class NNTrainer:

    def __init__(self, vector_model, embedding_size=100, scope_before=5, scope_after=5):
        self.vector_model = vector_model
        self.embedding_size = embedding_size
        self.scope_before = scope_before
        self.scope_after = scope_after

        self.dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.valid_dataset = None

    def add_train_samples(self, token, target):
        new_dataset = WindowDataset(token, target, self.vector_model, self.embedding_size,
                                    scope_before=self.scope_before, scope_after=self.scope_after)
        if self.dataset is None:
            self.dataset = new_dataset
        else:
            self.dataset = self.dataset + new_dataset
        

    def train(self, param):
        # Create the NN
        print("Training the RNN Modell: ", param)
        if param["save"]:
            model_path = os.path.join("Models", (param["log_file"][:-3] + "model2"))
        start_time = time.time()
        best_performance = {'f1': 0.0, 'recall': 0.0, 'precision': 0.0}
        lr = param["lr"]

        cpu = torch.device('cpu')
        if param["cuda"] and not torch.cuda.is_available():
            print("Using CPU")
            param["cuda"] = False

        message = ""
        for k, v in param.items():
            message += "{} {}\n".format(k, v)
        print(message)
        if param["log_file"]:
            with open(os.path.join("Logs", param["log_file"]), "a") as f:
                f.write(message)
                f.flush()

        """model = GruWindow(d_in=param["input_dim"], d_h=param["hidden_dim"], num_layers=param["layers"],
                           bidirectional=param["bidirectional"],
                           scope_before=param["scope_before"], scope_after=param["scope_after"])"""
        model = CNN()

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

            if param["lr_decay"] and not (e + 1) > 50 and e>0:
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

            f1 = 2 * true_pos / (
                    2 * true_pos + false_pos + false_neg) if 2 * true_pos + false_pos + false_neg > 0 else 0
            prec = true_pos / (true_pos + false_pos) if true_pos + false_pos > 0 else 0
            sens = true_pos / (true_pos + false_neg) if true_pos + false_neg > 0 else 0

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

        f1 = 2 * true_pos / (
                2 * true_pos + false_pos + false_neg) if 2 * true_pos + false_pos + false_neg > 0 else 0
        prec = true_pos / (true_pos + false_pos) if true_pos + false_pos > 0 else 0
        sens = true_pos / (true_pos + false_neg) if true_pos + false_neg > 0 else 0

        total = true_pos + true_neg + false_pos + false_neg
        pos = true_pos + false_neg
        neg = true_neg + false_pos
        acc = (true_pos + true_neg) / total if total > 0 else 0
        bacc = (true_pos / pos + true_neg / neg) / 2 if pos > 0 and neg > 0 else 0
        evaluation = {"recall": sens,
                      "precision": prec,
                      "acc": acc,
                      "bacc": bacc,
                      "f1": f1,}
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


class GruWindow(nn.Module):

    def __init__(self, d_in=100, d_h=20, num_layers=1, bidirectional=True, scope_before=7, scope_after=7):
        super(GruWindow, self).__init__()
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

class CNN(nn.Module):

    def __init__(self, d_in=100, d_h=20, num_layers=1, bidirectional=True, scope_before=3, scope_after=3):
        super(CNN, self).__init__()
        self.linear = nn.Linear(in_features=20, out_features=1)
        self.sig = nn.Sigmoid()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 5, 5, stride=2, padding=1),
            nn.Conv2d(5, 5, (3,5), stride=(1,2), padding=(1,0)),
            nn.Conv2d(5, 10, 5, stride=2),
            nn.Conv2d(10, 20, (2,10))
        )

    def forward(self, x):
        x = x[:,None,:,:]
        x = self.layers(x)
        x = torch.squeeze(x)
        x = self.linear(x)
        x = self.sig(x)
        return x


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
            token = vector_model[tok] if tok in vector_model else torch.zeros(embedding_size)
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
            token.append(torch.zeros(self.embedding_size, dtype=torch.float32))
        scope_start = max(scope_start, 0)
        for s in range(scope_start, idx):
            token.append(self.token[s])
        token.append(self.token[idx])
        for e in range(idx+1, min(self.__len__(), scope_end)):
            token.append(self.token[e])
        for e in range(self.__len__(), scope_end):
            token.append(torch.zeros(self.embedding_size, dtype=torch.float32))
        token = [torch.unsqueeze(t, dim=0) for t in token]
        output_token = torch.cat(token, dim=0)
        target = self.target[idx]

        return output_token, target

    def __len__(self):
        return len(self.token)

class PredictionModule:

    def __init__(self, vec_model_file="word2vec.wv", model_file="model2",
                 log_file="nn_ges.log", fetch_model=True, embedding_size=100,
                 self_trained_embeddings=True, scope_before=7, scope_after=7):
        self.vec_model_file = vec_model_file
        self.model_path = os.path.join(self.vec_model_file)
        self.nn_path = os.path.join(model_file)
        self.fetch_model = fetch_model
        self.log_file = log_file
        self.self_trained_embeddings = self_trained_embeddings
        self.word_model = None
        self.embedding_size = embedding_size
        self.scope_before = scope_before
        self.scope_after = scope_after
        self.trainer = None
        self.model = None
        self.param = {"input_dim": self.embedding_size, "hidden_dim": 64, "layers": 4, "bidirectional": True,
                      "save": True, "epochs": 40, "lr": 0.01, "lr_decay": True, "cuda": True,
                      "log_file": self.log_file, "seed": 42, "valid_percentage": 0.1, "test_percentage": 0.1,
                      "scope_before": self.scope_before, "scope_after": self.scope_after}

    def predict(self, token_list):
        """
        Predicts for every token whether it is considered a sentence boundary
        :param token_list: List of tokens
        :return: Annotations for every token on SBD
        """
        if self.word_model is None:
            self.load_model()
        if self.model is None and not self.fetch_model:
            self.finalize_training()
        elif self.model is None and self.fetch_model:
            self.model = torch.load(self.nn_path)
        self.model.eval()
        cpu = torch.device('cpu')
        self.model.to(cpu)
        fake_targets = [False]*len(token_list)
        dataset = WindowDataset(token_list, fake_targets, self.word_model, self.embedding_size,
                                scope_before=self.param["scope_before"], scope_after=self.param["scope_after"])
        dataloader = DataLoader(dataset, batch_size=1028)
        predictions = []
        for window, _ in dataloader:
            window = window.cuda()
            output = self.model.forward(window)
            output = output.cpu().detach().numpy()
            x = output.reshape(-1)
            ret = [True if i > 0.5 else False for i in x]
            for t in ret:
                predictions.append(t)
        return predictions

    def train(self, token, target=None):
        """
        If possible trains the classifier based on the token and target.
        :param token: The tokens which need to be classified.
        :param target: The groundtruth labels.
        :return: None
        """
        if self.word_model is None:
            self.load_model()
        if self.trainer is None:
            self.trainer = NNTrainer(self.word_model, self.embedding_size, scope_before=self.scope_before, scope_after=self.scope_after)
        self.trainer.add_train_samples(token, target)

    def finalize_training(self):
        self.model = self.trainer.train(self.param)
        print(self.model)

    def load_model(self):
        """
        Loads model.
        :return: None
        """
        if self.self_trained_embeddings:
            self.word_model = KeyedVectors.load(self.model_path, mmap='r')
        else:
            self.word_model = KeyedVectors.load_word2vec_format(self.model_path, binary=True)

def get_prediction(x):
    dim = x.shape
    x = x.reshape(-1)
    ret = np.array([1 if i > 0.5 else 0 for i in x])
    ret = ret.reshape(dim)
    return ret

filter_character = [' ', '\t', '\v', '\r', '\f']
alphanumeric = '[A-Za-z0-9ÄäÜüÖöß]'


def tokenize(input_directory):
    """
    Tokenizes the text found in an file. Uses different methods based on file extension. Possible file extensions are
    .txt and .json. When tokenizing .json files the annotations will also be extracted.
    :param input_directory: The directory/file to read the file from
    :return: dic with tokenlist as "Token" and annotations as "Labels"
    """
    labels = None
    if input_directory.endswith(".json"):
        with codecs.open(input_directory, 'r', 'utf-8') as f:
            dic = json.load(f)
        token, labels = tokenize_json(dic)
    else:
        raise ValueError("No possible input file. Use .json")
    return {"Token": token, "Labels": labels}


def tokenize_json(dic):
    """
    Tokenizes a json dic.
    :param dic: Dictionary containing the text ("Text") and the annotations ("Annotations")
    :return: Tokens and Annotations
    """
    token = []
    labels = []
    running_word = ""
    index = 0
    removed_characters = 0
    text = dic['Text']
    annotations = dic['Annotations']
    annotations.sort(key=lambda x: x["start"], reverse=True)
    anno_queue = deque(annotations)
    try:
        annotation = anno_queue.pop()
    except IndexError:
        annotation = None
    for c in text:
        if re.match(alphanumeric, c):
            running_word += c
            index += 1
        else:
            if running_word != "":
                token.append(running_word)
                if annotation is not None and annotation['start'] < index + removed_characters:
                    labels.append(True)
                    try:
                        annotation = anno_queue.pop()
                    except IndexError:
                        annotation = None
                else:
                    labels.append(False)
            if c in filter_character:
                removed_characters += 1
            else:
                token.append(c)
                if annotation is not None and annotation['start'] <= index + removed_characters:
                    labels.append(True)
                    try:
                        annotation = anno_queue.pop()
                    except IndexError:
                        annotation = None
                else:
                    labels.append(False)
                index += 1
            running_word = ""
    if running_word != "":
        token.append(running_word)
        if annotation is not None and annotation['start'] <= index + removed_characters:
            labels.append(True)
        else:
            labels.append(False)
    return token, labels

if __name__ == "__main__":
    print("Start NN Training")
    files = [file for file in os.listdir("../data") if file.endswith("json") and not file.startswith("xml")]
    print("Collected Files")
    predictionModule = PredictionModule(vec_model_file="word2vec_full.wv", log_file="model2", embedding_size=100,
                    self_trained_embeddings=True, scope_before=7, scope_after=7)
    print("Initialized PredictionModule")
    for file in files:
        print("Collecting:", file)
        train_dic = tokenize(os.path.join("../data", file))
        token = train_dic["Token"]
        labels = train_dic["Labels"]
        predictionModule.train(token, labels)
        print("Added file to training set")
    print("Start Optimization")
    predictionModule.finalize_training()
    print("Finished Optimization")
    model = torch.load("Models/model2")
    torch.save(model.state_dict(), "model2")
