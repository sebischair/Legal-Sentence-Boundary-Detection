# Trainer used for training the CRFModule
# Implements all the logic needed for training with multiple files etc.
from Modules import CRFModule
from Modules import calculateperformance
from FeatureExtractors import *
from CRFTrain import FeatureGenerator
import Tokenizer

import getopt
import sys
import os
import random
import math
import logging
import time

import pycrfsuite
from pycrfsuite import ItemSequence


def start_training(directory, text_type, save_location, verbose=False):
    """
    Starts the training procedure for the CRFModule
    :param directory: Path to all the training texts.
    :param text_type: Category of the text e.g. ges, agb or jug
    :param save_location: Folder location where to save the model
    :param verbose: Boolean flag whether the pycrfsuite Trainer should print additional information
    :return:
    """
    logger = logging.getLogger('training_logger')
    logger.setLevel(level=logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info("Started training")
    crf = CRFModule(model_path=save_location)
    # Collect all documents and their annotations -> method for generating the labels
    document_collection = []
    files = [f for f in os.listdir(directory) if f.endswith("json") and f.startswith(text_type)]
    for f in files:
        dic = Tokenizer.tokenize(os.path.join(directory, f))
        document_collection.append(dic)
    logger.info("Loaded files")
    # Processing the features for all documents
    feature_collection = []
    for document in document_collection:
        features = crf.generate_features(document['Token'])
        labels = [("SBD" if l else "NB") for l in document['Labels']]
        feature_collection.append({"Features": features, "Labels": labels, "Token": document["Token"]})
    logger.info("Processed features")
    # Get train/ test split for all the documents
    train_collection = []
    valid_collection = []
    test_collection = []
    for doc_feature in feature_collection:
        (train_dic, valid_dic) = create_train_test_split(doc_feature)
        (train_dic, test_dic) = create_train_test_split(train_dic)
        train_collection.append(train_dic)
        valid_collection.append(valid_dic)
        test_collection.append(test_dic)
    logger.info("Created train/test split")
    # Append train set to the Trainer
    trainer = pycrfsuite.Trainer(verbose=verbose)
    for doc in train_collection:
        trainer.append(ItemSequence(doc['Features']), doc['Labels'])
    # Start Trainer
    logger.info("Start trainer")
    trainer.train(save_location)
    logger.info("Finished trainer")
    # Evaluation on test set
    crf.load_model()
    y_true = []
    for doc in valid_collection:
        y_true = y_true + doc['Labels']
    y_pred = []
    for doc in valid_collection:
        y_pred = y_pred + crf.predict(doc['Token'])
    truth_true = []
    for true in y_true:
        if true == 'SBD':
            truth_true.append(True)
        else:
            truth_true.append(False)
    valid_performance = calculateperformance(truth_true, y_pred)
    y_true = []
    for doc in test_collection:
        y_true = y_true + doc['Labels']
    y_pred = []
    for doc in test_collection:
        y_pred = y_pred + crf.predict(doc['Token'])
    truth_true = []
    for true in y_true:
        if true == 'SBD':
            truth_true.append(True)
        else:
            truth_true.append(False)
    test_performance = calculateperformance(truth_true, y_pred)
    return valid_performance, test_performance


def create_train_test_split(doc_feature):
    """
    Splits the document up into a train and a test part.
    :param doc_feature: dictionary with "Labels", "Token" and "Features"
    :return:
    """
    test_percentage = 0.3
    start_index = random.randint(1, len(doc_feature["Token"])-1)
    size = math.ceil(test_percentage * len(doc_feature["Token"]))
    end_index = (start_index + size) % len(doc_feature["Token"])
    train_features = {}
    test_features = {}
    if end_index < start_index:
        for lab in ["Labels", "Token", "Features"]:
            test_features[lab] = doc_feature[lab][start_index:]+doc_feature[lab][:end_index]
            train_features[lab] = doc_feature[lab][end_index:start_index]
    else:
        for lab in ["Labels", "Token", "Features"]:
            test_features[lab] = doc_feature[lab][start_index:end_index]
            train_features[lab] = doc_feature[lab][end_index:]+doc_feature[lab][:start_index]
    return train_features, test_features


def feature_exploration(directory, save_directory, text_type, feature_generator, log_file="test.log", runs=5):
    """
    Searches for the best features in a grid style.
    :param directory: Where to find the training files
    :param save_directory: Where to save the model
    :param text_type: Which document beginnings to choose
    :param feature_generator: A generator for all the features
    :param log_file: Where to save the log
    :return:
    """
    start_time = time.time()
    logger = logging.getLogger('exploration_logger')
    logger.setLevel(level=logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info("Time: " + str(time.asctime(time.localtime(start_time))))
    logger.info("Started exploration")
    # Get document collection
    document_collection = []
    files = [f for f in os.listdir(directory) if f.endswith("json") and f.startswith(text_type)]
    for f in files:
        dic = Tokenizer.tokenize(os.path.join(directory, f))
        document_collection.append(dic)
    logger.info("Loaded files")
    logger.addHandler(logging.FileHandler(os.path.join("Logs", log_file)))
    extractors = feature_generator.generate_feature_combinations()
    for ex in extractors:
        round_start_time = time.time()
        total_performance = None
        total_test_performance = None
        special_path = generate_special_path(text_type, ex)
        crf = CRFModule(save_directory=save_directory, model_path=special_path)
        crf.feature_extractors = ex
        # Processing the features for all documents
        feature_collection = []
        for document in document_collection:
            features = crf.generate_features(document['Token'])
            labels = [("SBD" if l else "NB") for l in document['Labels']]
            feature_collection.append({"Features": features, "Labels": labels, "Token": document["Token"]})
        logger.info("Started Training on: %s" % special_path)
        logger.info("Number of runs: " + str(runs))
        # Multiple runs for averaging
        for r in range(runs):
            # Get train/ test split for all the documents
            train_collection = []
            valid_collection = []
            test_collection = []
            for doc_feature in feature_collection:
                (train_dic, valid_dic) = create_train_test_split(doc_feature)
                (train_dic, test_dic) = create_train_test_split(train_dic)
                train_collection.append(train_dic)
                valid_collection.append(valid_dic)
                test_collection.append(test_dic)
            # Append train set to the Trainer
            trainer = pycrfsuite.Trainer(verbose=False)
            for doc in train_collection:
                trainer.append(ItemSequence(doc['Features']), doc['Labels'])
            # Start Trainer
            trainer.train(os.path.join(save_directory, special_path))

            # Evaluation on test set
            crf.load_model()
            y_true = []
            for doc in valid_collection:
                y_true = y_true + doc['Labels']
            y_pred = []
            for doc in valid_collection:
                y_pred = y_pred + crf.predict(doc['Token'])
            truth_true = []
            for true in y_true:
                if true == 'SBD':
                    truth_true.append(True)
                else:
                    truth_true.append(False)
            valid_performance = calculateperformance(truth_true, y_pred)
            y_true = []
            for doc in test_collection:
                y_true = y_true + doc['Labels']
            y_pred = []
            for doc in test_collection:
                y_pred = y_pred + crf.predict(doc['Token'])
            truth_true = []
            for true in y_true:
                if true == 'SBD':
                    truth_true.append(True)
                else:
                    truth_true.append(False)
            test_performance = calculateperformance(truth_true, y_pred)

            if total_performance is None:
                total_performance = valid_performance
            else:
                for k in valid_performance:
                    total_performance[k] += valid_performance[k]
            if total_test_performance is None:
                total_test_performance = test_performance
            else:
                for k in test_performance:
                    total_test_performance[k] += test_performance[k]
        # Save average performance
        for k in total_performance:
            logger.info("%s: %1.4f" % (k, total_performance[k]/runs))
            logger.info("Test, %s: %1.4f" % (k, total_test_performance[k]/runs))
        round_time = time.time() - round_start_time
        str_round = time.strftime("%M.%Smin", time.gmtime(round_time))
        logger.info("Round time: " + str_round)
    end_time = time.time()
    duration = end_time - start_time
    str_dur = time.strftime("%jd %Hh %Mmin %Ssec", time.gmtime(duration))
    logger.info("Total time: " + str_dur + "min")


def generate_special_path(text_type, ex):
    return text_type + ''.join(["_"+f.name+"("+str(f.scope_before)+","+str(f.scope_after)+")" for f in ex])


if __name__ == "__main__":
    argv = sys.argv[1:]
    directory = '../data'
    text_type = 'ges'
    save_location = 'Models'
    try:
        opts, args = getopt.getopt(argv, "hi:o:f:", ["indir=", "output=", "text_type="])
    except getopt.GetoptError:
        print("CRFTrainer.py -i <input_directory> -o <output> -f <text_type>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("CRFTrainer.py -i <input_directory> -o <output> -f <text_type>")
        elif opt in ("-i", "--indir"):
            directory = arg
        elif opt in ("-o", "--output"):
            save_location = arg
        elif opt in ("-f", "--text_type"):
            text_type = arg
    save_location = save_location + '\\crf_' + text_type + '_model.mdl'
    start_training(directory, text_type, save_location)
