from Modules import NLTKModule

import sys
import os
import getopt
import codecs

if __name__ == "__main__":
    argv = sys.argv[1:]
    t_d = '../data'
    try:
        opts, args = getopt.getopt(argv, "ht:", ["traindir="])
    except getopt.GetoptError:
        print("NLTKTrainer.py -t <train_directory>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("NLTKTrainer.py -t <train_directory>")
        elif opt in ("-t", "--traindir"):
            t_d = arg
    model = NLTKModule()
    model.model_path = "Models/sbd_model.mdl"
    files = [f for f in os.listdir(t_d) if f.endswith("txt")]
    for f in files:
        print("Training on: ", f)
        with codecs.open(os.path.join(t_d, f), 'r', 'utf-8') as s:
            text = s.read()
        model.train_text(text)
    model.finalize_training()
