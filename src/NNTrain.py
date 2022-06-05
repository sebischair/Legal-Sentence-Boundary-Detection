from NNTrainer import *
from Modules import NNModule
import Tokenizer

import os

type = "jug"
type2 = "ges"

if __name__ == "__main__":
    for i in range(1, 6):
        print("Start NN Training")
        files = [file for file in os.listdir("../data") if file.endswith("json")]
        print("Collected Files")
        nn = NNModule(vec_model_file="word2vec_full.wv", log_file="p_total_gru22_nn"+str(i)+".log", embedding_size=100,
                      self_trained_embeddings=True, scope_before=7, scope_after=7)
        print("Initialized Module")
        for file in files:
            print("Collecting:", file)
            train_dic = Tokenizer.tokenize(os.path.join("../data", file))
            token = train_dic["Token"]
            labels = train_dic["Labels"]
            nn.train(token, labels)
            print("Added file to training set")
        print("Start Optimization")
        nn.finalize_training()
        print("Finished Optimization")
