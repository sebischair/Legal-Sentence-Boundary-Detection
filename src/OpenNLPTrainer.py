import Tokenizer
import Splitter
from Modules import OpenNLPModule

import os
import codecs
import json


# Decides which files to use for training
file_type = "agb"
text_path = "../data"
print("Train on:", file_type)
files = [f for f in os.listdir(text_path) if f.startswith(file_type) and f.endswith("json")]
opennlp = OpenNLPModule(model_file=file_type + "_open_nlp.bin", train_file=file_type + ".txt")
# Collect all files
print("Started collecting files")
for file in files:
    path = os.path.join(text_path, file)
    print(path)
    with codecs.open(path, "r", "utf-8") as f:
        dic = json.load(f)
    text = dic["Text"]
    token_dic = Tokenizer.tokenize(path)
    sentences = Splitter.split_sentences(text, token_dic["Labels"])
    opennlp.train_text(sentences)
# train_text
print("Finalize training")
# finalize_training
opennlp.finalize_training()
