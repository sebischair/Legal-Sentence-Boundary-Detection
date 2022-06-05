import Tokenizer

import re
import pickle
import os
import codecs
import subprocess
import sys
from pprint import pprint

from collections import deque
import math
from functools import partial

# Imports used for NLTKModule
import nltk.tokenize.punkt as punkt
from nltk.tokenize import PunktSentenceTokenizer
import nltk.data

# Imports used for CRFModule
import pycrfsuite
from FeatureExtractors import *

import torch
from torch.utils.data import DataLoader

from NNTrainer import WindowDataset

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

from NNTrainer import NNTrainer


class SBDModule:

    def __init__(self):
        """
        Sets up the module.
        """
        # A variable which helps to distinguish methods based on whether they predict
        # given tokens or text
        self.predict_on_token = True
        self.trainable = False

    def predict(self, token_list):
        """
        Predicts for every token whether it is considered a sentence boundary
        :param token_list: List of tokens
        :return: Annotations for every token on SBD
        """
        return [False] * len(token_list)

    def predict_text(self, text):
        """
        Returns prediction for every token wheter it is considered a sentence boundary.
        Decision based on a String input.
        :param text: String
        :return: Annotations for every token on SBD
        """
        return self.predict(Tokenizer.tokenize_text(text))

    def train(self, token, target=None):
        """
        If possible trains the classifier based on the token and target.
        :param token: The tokens which need to be classified.
        :param target: The groundtruth labels.
        :return: None
        """
        # raise NotImplementedError()
        pass

    def train_text(self, text):
        """
        Trains the classifier based on the text. This is mainly used for unsupervised methods.
        :param text: The input text to train on
        :return:
        """
        # raise NotImplementedError()
        pass

    def load_model(self):
        """
        Loads model.
        :return: None
        """
        # raise NotImplementedError()
        pass


class RuleModule(SBDModule):

    def __init__(self):
        """
        The window rules are the rules needed to assign a sentence boundary. When assigning the SB the rules will be
        ordered by window size and then every rule for the possible window size applied, positive before negative.
        The window size of a rule can be seen in the length of the tuple corresponding to that rule.
        The ceil of windowsize/2 will be the possition for the possible sentence boundary.
        """
        SBDModule.__init__(self)
        # TODO do we really need raw strings?
        self.alphaNumeric = r"[0-9A-Za-zÄäÜüÖöß]+"
        self.upper = r"[A-ZÄÜÖ][0-9A-Za-zÄäÜüÖöß]*"
        self.lower = r"[a-zäüö]+"
        self.number = r"[0-9]+"
        self.character = r"^[A-Za-zÄäÜüÖöß]\Z"
        self.consonant = r"^([QqWwEeRrTtZzPpSsDdFfGgHhJjKkLlYyXxCcVvBbNnMm]+)\Z"
        self.abreviation = r"^[A-Za-zÄäÜüÖöß]{1,3}\Z"
        self.abreviation_list = r"Abs|Art|Rn|Urt|Buchst|bzw|usw|Bl|ff|Nr|ca|Hr"
        self.everything = ".+"

        self.sentence_boundary_character = r"[.:;?!]"
        self.headline_boundary_character = r"[:\)!]"
        self.parentheses = r"[\(\)\[\]\{\}]"
        self.special = r"[.:;?!,#*§$%&()\[\]\{\}]"
        self.dot = r"\."
        self.comma = r","
        self.paragraph = r"§|Art"
        self.new_content = self.paragraph + "|" + self.number
        self.newline = r"\n"
        self.citation = r"\""
        self.positive_rules = [
            # Normal sentence
            [self.alphaNumeric, self.alphaNumeric, self.alphaNumeric, self.sentence_boundary_character, self.upper],
            # End of Citations
            [self.alphaNumeric, self.alphaNumeric, self.alphaNumeric, self.sentence_boundary_character, self.citation],
            [self.alphaNumeric, self.alphaNumeric, self.alphaNumeric, self.citation, self.sentence_boundary_character],
            # End of line
            [self.everything, self.everything, self.alphaNumeric, self.sentence_boundary_character, self.newline],
            # Paragraph start
            [self.paragraph + "|" + self.upper, self.number, self.alphaNumeric, self.newline],
            [self.paragraph, self.number, self.newline],
            # Headline
            [self.newline, self.alphaNumeric, self.newline],
            [self.upper, self.upper, self.newline],
            [self.everything, self.everything, self.upper, self.newline, self.number + "|" + self.paragraph],
            # Before new point
            [self.everything, self.everything, self.everything, self.alphaNumeric, self.sentence_boundary_character,
                self.newline, self.new_content],
            [self.comma, self.alphaNumeric, self.dot],
            # Sentence
            [self.alphaNumeric, self.alphaNumeric, self.sentence_boundary_character],
            [self.newline, self.alphaNumeric, self.sentence_boundary_character],
            [self.alphaNumeric, self.alphaNumeric, self.alphaNumeric, self.sentence_boundary_character, self.upper],
            [self.everything, self.newline, self.alphaNumeric, self.sentence_boundary_character, self.newline],
            # After abreviations
            [self.abreviation, self.dot, self.alphaNumeric, self.sentence_boundary_character],
            # After parentheses
            [self.everything, self.parentheses, self.sentence_boundary_character],
            # Page numbers
            [self.alphaNumeric, self.number, r"-", self.newline, self.special+"|"+self.paragraph]
        ]
        self.negative_rules = [
            # Abbreviations
            [self.alphaNumeric, self.alphaNumeric, self.abreviation, self.dot, self.alphaNumeric],
            # Number
            [self.everything, self.number, self.dot],
            # No Headline
            [self.everything, self.everything, self.alphaNumeric, self.newline, self.lower],
            # Sentence end characters
            [self.sentence_boundary_character],
            # Enumeration start
            [self.newline, self.number, self.dot],
            [self.newline, self.special, self.number, self.special, self.upper],
            # Single Character/ Abreviation
            [self.everything + "|" + self.newline, self.character, self.dot],
            [self.abreviation_list]
            # TODO do not work that well
            #[self.everything, self.consonant, self.dot]
            #[self.everything, self.everything, self.alphaNumeric, self.dot, self.lower]
        ]

    def predict(self, token_list):
        """
        Predicts for every token whether it is considered a sentence boundary
        :param token_list: List of tokens
        :return: Annotations for every token on SBD
        """
        prediction = [False] * len(token_list)
        max_window = max(max(len(rule) for rule in self.positive_rules), max(len(rule) for rule in self.negative_rules))
        # Runs the positive rules over the whole windowed text
        for window_size in range(1, max_window+1):
            pos_rules = [rule for rule in self.positive_rules if len(rule) == window_size]
            win_gen = window_generator(token_list, window_size)
            if len(pos_rules) > 0:
                for i, window in enumerate(win_gen):
                    yes = self.apply_rules(pos_rules, window)
                    prediction[i] = prediction[i] or yes

        # Runs the negation rule over the whole windowed text
        for window_size in range(1, max_window+1):
            neg_rules = [rule for rule in self.negative_rules if len(rule) == window_size]
            win_gen = window_generator(token_list, window_size)
            if len(neg_rules) > 0:
                for i, window in enumerate(win_gen):
                    no = self.apply_rules(neg_rules, window)
                    prediction[i] = prediction[i] and not no
        return prediction

    def apply_rules(self, rule_list, window):
        """
        Checks whether a rule in the rule_list applies to the given window.
        :param rule_list: List of different rules.
        :param window: Window of tokens
        :return: True: One rule matched, ow False
        """
        reduce = partial(self.reduce_rule, len(window))
        rule_list = map(reduce, rule_list)
        sentence_boundary = False
        for rule in rule_list:
            match = True
            for i, token in enumerate(window):
                if not re.match(rule[i], token):
                    match = False
                    break
            if match:
                sentence_boundary = True
                break
        return sentence_boundary

    def reduce_rule(self, window_size, rule):
        """
        Used when the rule is to big for the window. Shortens the given rule.
        :param window_size: The window size
        :param rule: A given rule
        :return: Rule with the same length as the window size
        """
        return rule[-window_size:]


class TemplateModule(SBDModule):

    def __init__(self, fetch_module=False, annotator=None):
        """
        Sets up the module.
        """
        SBDModule.__init__(self)

        self.predict_on_token = False
        self.annotator = annotator

        self.model_path = "Models\\template_model.mdl"
        self.trigger_list = []
        self.body_trigger = r""
        self.word = r"[A-Za-zÄäÜüÖöß]+"
        self.character = r"[A-Za-zÄäÜüÖöß]"
        self.alpanumeric = r"[0-9A-Za-zÄäÜüÖöß]+"
        self.number = r"[0-9]+"
        self.headline_boundary_character = [".", "," ":", "!", "?"]

        self.split_on_headlines = False
        self.max_triggers = 2

        self.trainable = True

        if fetch_module:
            with open(self.model_path, mode='rb') as f:
                self.trigger_list, self.body_trigger = pickle.load(f)
            self.trainable = False

    def predict(self, token_list):
        """
        Predicts the
        :param token_list: List of tokens
        :return: Annotations for every token on SBD
        """
        pass

    def predict_text(self, text):
        """
        Returns prediction for every token wheter it is considered a sentence boundary.
        Decision based on a String input.
        :param text: String
        :param split_on_headlines: Boolean value, whether headlines need to be split from following text. Usefull
        when the following text parts are not numerically annotated. Only use this if headline is always followed by
        text or headlines do not contain abreviations. Otherwise there might be a wrong split.
        :return: Annotations for every token on SBD
        """
        # TODO Segment the text into head and body
        segment_list = [text]
        # Produce the segments based on the trigger tokens, each token resembles one split
        for trigger in self.trigger_list:
            new_segments = []
            # Every segment possibly consists of more subsegments
            for segment in segment_list:
                regex = re.compile(trigger)
                # We do not want to produce a split at the beginning of the sentence, thus pos = 1 for regex
                match = regex.search(segment, 1)
                while match is not None and match.start() < match.end():
                    start = match.start()
                    # The trigger is always based on the beginning of the line that means if we encounter a newline
                    # it should be part of the previous segment.
                    if segment[start] == '\n':
                        start += 1
                    new_segments.append(segment[:start])
                    segment = segment[start:]
                    match = regex.search(segment, 1)
                # If there is a segment at the end, it needs to be added
                if len(segment) > 0:
                    new_segments.append(segment)
            segment_list = new_segments
        # Extract headline from standard text:
        if self.split_on_headlines:
            segment_list = self.split_headlines(segment_list)
        if self.annotator is not None:
            prediction = []
            for segment in segment_list:
                seg_prediction = self.annotator.predict_text(segment)
                for pred in seg_prediction:
                    prediction.append(pred)
        else:
            prediction = prediction_from_sentences(segment_list)
        return prediction

    def split_headlines(self, segment_list):
        new_segments = []
        for segment in segment_list:
            line, length = self.extract_line(segment)
            # Is the first line the whole segment?
            if len(line) < len(segment):
                line += '\n'
                sub_segment = segment[length:]
                last_cut = 0
                encountered_headline_terminating_character = False
                # Subsegments can b
                for i, c in enumerate(sub_segment):
                    if c == "\n" and not encountered_headline_terminating_character:
                        line += sub_segment[last_cut:i + 1]
                        last_cut = i + 1
                    elif c in self.headline_boundary_character:
                        encountered_headline_terminating_character = True
                        break
                # Whether a subsegment could be found
                if encountered_headline_terminating_character:
                    new_segments.append(line)
                    new_segments.append(sub_segment[last_cut:])
                else:
                    new_segments.append(segment)
            else:
                new_segments.append(segment)
        return new_segments

    def train(self, token, target=None):
        """
        Train the template based on the tokens. Target is not taken into consideration. The trigger list is constructed
        by looking at recurrent patterns in the text. That mean one pattern at the start of a line, that is in some form
        special.
        :param token: The tokens which need to be classified.
        :param target: The groundtruth labels.
        :param max_triggers: Maximum number of triggers in the trigger  list. Keeps the number of substructures down.
        :return: None
        """
        # Get first encounter with numeric pattern starting at 1
        # TODO Look at \r???
        start_body = 0
        last_newline = 0
        for i, tok in enumerate(token):
            if tok == '\n':
                last_newline = 0
            else:
                if tok == '1' and last_newline <= 3:
                    start_body = i-last_newline
                    break
                else:
                    last_newline += 1
        body = token[start_body:]
        # Extraction of the body trigger
        line, _ = self.extract_line(body)
        body_trigger = self.extract_trigger(line)
        trigger_list = [body_trigger]
        # Split up the segments, if a new split is made the result is appended to the segment list
        segment_list = self.segment(body_trigger, body)
        segment_queue = self.get_segment_queue(segment_list)
        for segment in segment_queue:
            _, length = self.extract_line(segment)
            line, _ = self.extract_line(segment[length:])
            trigger = self.extract_trigger(line)
            if len(trigger) > 1 and len(trigger_list) < self.max_triggers:
                new_segments = self.segment(trigger, segment)
                if len(new_segments) > 1:
                    for new_segment in new_segments:
                        segment_list.append(new_segment)
                    if trigger not in trigger_list:
                        trigger_list.append(trigger)
        # The found trigger lists need to be translated to regex
        self.trigger_list = []
        self.body_trigger = self.translate_trigger_from_list(body_trigger)
        for trigger in trigger_list:
            self.trigger_list.append(self.translate_trigger_from_list(trigger))
        # Save the trigger lists to disk
        if self.trainable and False:
            with open(self.model_path, mode='wb') as f:
                trigger = (self.trigger_list, self.body_trigger)
                pickle.dump(trigger, f, protocol=pickle.HIGHEST_PROTOCOL)

    def train_text(self, text):
        """
        Trains the classifier based on the text. This is mainly used for unsupervised methods.
        :param text: The input text to train on
        :return: None
        """
        token = Tokenizer.tokenize_text(text)
        self.train(token)

    def define_triggers(self, trigger_list, body_trigger="", path=None):
        """
        Sets the trigger_list to the wanted structures. This trigger list is then saved at the path.
        :param trigger_list:
        :param body_trigger: Trigger used to destinguish the head and the body of a text.
        :param path: Path for saved model. If none is given, triggers will be saved to default path.
        :return:
        """
        self.trigger_list = trigger_list
        self.body_trigger = body_trigger
        if path is None:
            path = self.model_path
        else:
            self.model_path = path
        with open(path, mode='wb') as f:
            trigger = (self.trigger_list, self.body_trigger)
            pickle.dump(trigger, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, path=None):
        """
        Loads model. Possibility of adding a path to the model. If not given loads from the default path.
        :param path: Path to load the model from. If no path is given the model will be loaded form the predefined path.
        :return: None
        """
        if path is None:
            path = self.model_path
        with open(path, "rb") as f:
            self.trigger_list, self.body_trigger = pickle.load(f)

    def extract_line(self, token):
        """
        Extracts the first line from a given token list
        :param token: List of tokens
        :return: First line of the token list
        """
        index_newline = 0
        for tok in token:
            if tok != '\n':
                index_newline += 1
            else:
                break
        token_length_tuple = (token[:index_newline], index_newline+1)
        return token_length_tuple

    def extract_trigger(self, token):
        """
        Method extracts a universal regex to match a found numeric pattern at the beginning of a new line.
        :param token: Token list to extract trigger list from
        :return: Trigger for every individual token (if numeric expression was found, else empty trigger list)
        """
        end_trigger_area = 0
        encounterd_numeric = False
        for tok in token:
            if encounterd_numeric:
                if re.match(self.word+r"|\n", tok) and len(tok) > 1:
                    break
                end_trigger_area += 1
            else:
                end_trigger_area += 1
                if tok == '1':
                    encounterd_numeric = True
                elif tok == '\n':
                    break
        trigger_area = token[:end_trigger_area]
        trigger_list = []
        for tok in trigger_area:
            if re.match(self.number, tok):
                trigger_list.append(self.number)
            elif re.match(self.alpanumeric, tok):
                trigger_list.append(tok)
            else:
                trigger_list.append(re.escape(tok))
        trigger_list.insert(0, r"\n")
        trigger = trigger_list
        if not encounterd_numeric:
            trigger = []
        return trigger

    def translate_trigger_from_list(self, trigger_list):
        """
        Converts a trigger list into a regular expression
        :param trigger_list: trigger list to be converted
        :return: regular expression
        """
        regex_list = []
        encountered_special_character = False
        for trigger in trigger_list:
            if trigger == r'\n':
                regex_list.append(trigger)
                encountered_special_character = True
            elif self.number == trigger:
                if not encountered_special_character:
                    regex_list.append(" ")
                regex_list.append(trigger)
                encountered_special_character = False
            elif self.alpanumeric == trigger or re.match(self.alpanumeric, trigger):
                if len(trigger) > 2 and not encountered_special_character:
                    regex_list.append(" ")
                regex_list.append(trigger)
                encountered_special_character = False
            else:
                regex_list.append(trigger)
                encountered_special_character = True
        return ''.join(regex_list)

    def segment(self, trigger, segment):
        """
        Builds the subsegments of a segment based on a given trigger.
        :param trigger: The trigger used to determine the split indices
        :param segment: The segment (token list) to be split up into subsegments, if the trigger matches
        :return: Segment list of all generated subsegments (if no segments where found return list with original segm.)
        """
        # Newline is part of the first line -> thus last_break the index for the token after the last split is 1
        last_break = 0
        new_segments = []
        for i in range(len(segment)):
            if len(trigger)+i <= len(segment):
                match = True
                for n, t in enumerate(trigger):
                    if not re.match(t, segment[i+n]):
                        match = False
                        break
                if match:
                    new_segments.append(segment[last_break:i+1])
                    last_break = i+1
        if len(new_segments) == 0:
            new_segments.append(segment)
        if len(segment[last_break:]) > 0 and len(new_segments) > 1:
            new_segments.append(segment[last_break:])
        return new_segments

    def get_segment_queue(self, segment_list):
        """
        Builds a dynamic queue, that can be used in for loops. It is still possible to append segments to the end of
        segment_list. Those will be also iterated over.
        :param segment_list: Segment list to iterate over
        :return: List Generator
        """
        index = 0
        while index < len(segment_list):
            yield segment_list[index]
            index += 1

    def set_split_on_headlines(self, split=True):
        self.split_on_headlines = split


class CRFModule(SBDModule):

    def __init__(self, save_directory="Models", model_path="crf_ges_model.mdl", feat_ex=None):
        # A variable which helps to distinguish methods based on whether they predict
        # given tokens or text
        SBDModule.__init__(self)
        self.model_path = os.path.join(save_directory, model_path)
        self.tagger = None
        if feat_ex is None:
            # TODO Load from model name
            self.feature_extractors = [FeatureExtractor(), LengthExtractor(scope_before=2, scope_after=2),
                                       UpperExtractor(scope_after=1),
                                       SpecialExtractor(scope_after=1, on_token=False)]
        else:
            self.feature_extractors = feat_ex

    def predict(self, token_list):
        """
        Predicts for every token whether it is considered a sentence boundary
        :param token_list: List of tokens
        :return: Annotations for every token on SBD
        """
        # Process token_list to features
        if self.tagger is None:
            self.load_model()
        features = self.generate_features(token_list)
        labels = self.tagger.tag(features)
        prediction = []
        for label in labels:
            if label == "SBD":
                prediction.append(True)
            else:
                prediction.append(False)
        return prediction

    def train(self, token, target=None):
        """
        If possible trains the classifier based on the token and target.
        :param token: The tokens which need to be classified.
        :param target: The groundtruth labels.
        :return: None
        """
        pass

    def load_model(self):
        """
        Loads model.
        :return: None
        """
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(self.model_path)

    def generate_features(self, token_list):
        feature_dic = []
        for extractor in self.feature_extractors:
            feature_dic = extractor.process_feature(token_list, feature_dic)
        return feature_dic


class NNModule(SBDModule):

    def __init__(self, vec_model_file="word2vec.wv", model_file="ges_lstm_2layer_2linear_wcW.model",
                 log_file="nn_ges.log", fetch_model=True, embedding_size=100,
                 self_trained_embeddings=True, scope_before=7, scope_after=7):
        SBDModule.__init__(self)
        self.vec_model_file = vec_model_file
        self.model_path = os.path.join("..", "data", self.vec_model_file)
        self.nn_path = os.path.join("..", "Models", model_file)
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
                      "save": True, "epochs": 60, "lr": 0.00025, "lr_decay": True, "cuda": True,
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
        #self.model.to(cpu)
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
            self.trainer = NNTrainer(token, target, self.word_model, self.embedding_size,
                                     scope_before=self.scope_before,
                                     scope_after=self.scope_after)
        else:
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

    def generate_word_vectors(self, path="../data", files=["ges.txt", "jug.txt", "agb.txt"], vec_size=100):
        # Convert files to iterables
        # Generate dictionary
        texts = []
        for file in files:
            print(file)
            file_path = os.path.join(path, file)
            with codecs.open(file_path, "r", "utf-8") as f:
                text = f.read()
            documents = text.split("\n\n")
            for doc in documents:
                if len(doc) > 3:
                    texts.append(doc)
        print("Num. Documents:", len(texts))
        print("Tokenize Documents")
        texts = [Tokenizer.tokenize_text(doc) for doc in texts]
        print("Start Training")
        model = Word2Vec(texts, size=vec_size, window=5, min_count=2)
        print("Finished Training")
        print(model)
        model.wv.save(os.path.join(path, "word2vec_full.wv"))
        print("WordVectors saved")


class NLTKModule(SBDModule):

    def __init__(self, fetch_module=False, model_path="../Models/sbd_model.mdl"):
        """
        Intializes the NLTK Punkt module, which is then used to
        """
        SBDModule.__init__(self)

        self.special = r"[\.,;:*#!?\"\(\{\}\[\]-_]\Z"

        # Can be used to see which prediction method needs to be used
        # NLTK only works on text
        self.predict_on_token = False

        self.fetch_model = fetch_module

        self.in_training = False
        self.trainer = None

        self.model_path = model_path
        if self.fetch_model:
            self.load_model()
        else:
            self.model = nltk.data.load('tokenizers/punkt/german.pickle')

    def predict_text(self, text):
        """
        Predicts for every token whether it is considered a sentence boundary
        :param text: Prediction also possible based on text (needs to be tokenized)
        :return: Annotations for every token on SBD
        """
        if self.in_training:
            self.finalize_training()
        sentences = self.model.tokenize(text)
        # Queue for token, so it is possible to track whether a newline
        # was omitted at the end of the sentence. NLTK removes newlines when
        # they are at the end of the sentence.
        token = Tokenizer.tokenize_text_with_special(text)
        index = 0
        processed_sentences = []
        for sentence in sentences:
            token_sentence = []
            missing_token = 0
            for i, tok in enumerate(Tokenizer.tokenize_text_with_special(sentence), index):
                token_sentence.append(tok)
                if tok != token[i+missing_token]:
                    token_sentence.append(token[i+missing_token])
                    index += 1
                    missing_token += 1
                index += 1
            while len(token) > index and token[index] in ['\n', ' ', '\r', '\t']:
                token_sentence.append(token[index])
                index += 1
            processed_sentences.append(''.join(token_sentence))
        # Now every missing token is inserted to the position it was omitted by NLTK
        # Now it is possible to create the predictions.
        for t, p in zip(text, ''.join(processed_sentences)):
            try:
                assert(t == p)
            except:
                print(t, p)
                exit(-1)
        sentences = processed_sentences
        prediction = prediction_from_sentences(sentences, self.special)
        return prediction

    def train_text(self, text):
        """
        Trains the Punkt classifier based on the String.
        :param text: Training String
        :return: None
        """
        if not self.fetch_model:
            self.in_training = True
            self.trainer = punkt.PunktTrainer()
        self.trainer.train(text, finalize=False)

    def finalize_training(self):
        """
        Finalizes the training and converts the trainer to a model.
        :return: None
        """
        self.trainer.finalize_training()
        self.fetch_model = True
        self.in_training = False
        self.model = PunktSentenceTokenizer(self.trainer.get_params())
        self.trainer = None
        with open(self.model_path, mode='wb') as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self):
        """
        Loads model from path.
        :return: None
        """
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
            self.fetch_model = True


class OpenNLPModule(SBDModule):

    def __init__(self, model_file="jug_open_nlp.bin", train_file="sent.txt",
                 collection_path="collection.txt", predict_path="prediction.txt"):
        """
        Sets up the module. This whole module works by communicating with the Java implementation of corenlp.
        Communication is file based so no integration is needed in this case.
        """
        SBDModule.__init__(self)

        self.special = r"[\.,;:*#!?\"\(\{\}\[\]-_]\Z"

        self.predict_on_token = False
        self.trainable = False
        self.path = os.path.join("Models", model_file)
        self.train_path = os.path.join("../data", train_file)
        self.predict_path = os.path.join("Models", predict_path)
        self.collection_path = os.path.join("Models", collection_path)
        self.sentences = []

    def predict(self, token_list):
        """
        Predicts for every token whether it is considered a sentence boundary
        :param token_list: List of tokens
        :return: Annotations for every token on SBD
        """
        pass

    def predict_text(self, text):
        """
        Returns prediction for every token wheter it is considered a sentence boundary.
        Decision based on a String input.
        :param text: String
        :return: Annotations for every token on SBD
        """
        # move text to prediction file
        new_text = []
        for c in text:
            if c == "\n":
                new_text.append(" ")
            elif c != "\r":
                new_text.append(c)
        new_text = ''.join(new_text)
        with codecs.open(self.collection_path, "w+", "utf-8") as f:
            f.write(new_text)
        # start opennlp
        subprocess.run(["opennlp", "SentenceDetector", self.path, "<", self.collection_path, ">", self.predict_path], shell=True)
        # retrieve lines from prediction file
        pred_text = []
        with codecs.open(self.predict_path, "r", "utf-8") as f:
            for line in f:
                # Remove newlines and carriage returns from the predicted sentences
                if line.endswith("\n"):
                    line = line[:-1]
                if line.endswith("\r"):
                    line = line[:-1]
                pred_text.append(line)
        pred_text = [Tokenizer.tokenize_text_with_special(sent) for sent in pred_text]
        token = Tokenizer.tokenize_text_with_special(text)
        index = 0
        processed_sentences = []
        # For every sentence check whether some tokens are missing
        for i, sentence in enumerate(pred_text):
            # Run over every individual token
            running_sentence = []
            for tok_index in range(len(sentence)):
                # If it is the same at token[index] (the original text) just append and inc the index
                sent_tok = sentence[tok_index]
                real_tok = token[index]
                if index >= len(token) or sent_tok == real_tok or (sent_tok == " " and real_tok == "\n"):
                    running_sentence.append(real_tok)
                    index += 1
                # Else you append the tokens until you find the next token or the first token in the next sentence
                else:
                    if tok_index < len(sentence)-1:
                        next_token = sentence[tok_index]
                        append_next = True
                    elif i < len(pred_text)-1:
                        try:
                            assert len(pred_text[i+1]) > 0
                            next_token = pred_text[i+1][0]
                            append_next = False
                        except AssertionError:
                            print(pred_text[i])
                    else:
                        next_token = None
                        append_next = False
                    while index < len(token) and token[index] != next_token:
                        running_sentence.append(token[index])
                        index += 1
                    if append_next:
                        running_sentence.append(next_token)
                        index += 1
            processed_sentences.append(''.join(running_sentence))
        # Now every missing token is inserted to the right position
        # Now it is possible to create the predictions.
        for t, p in zip(text, ''.join(processed_sentences)):
            assert (t == p)
        # generate predictions based on lines
        sentences = processed_sentences
        prediction = prediction_from_sentences(sentences, self.special)
        return prediction

    def train_text(self, text):
        """
        Trains the classifier based on the text. This is mainly used for unsupervised methods. Accepts input text in the
        following format: each sentence corresponds to one line in the text. When finalize_training is called the string
        will be written to the data.train location. This file is the direct input of the training API, which will then
        be called.
        :param text: The input text to train on, not tokenized, but already split up into the sentences
        :return: None
        """
        for line in text:
            self.sentences.append(line)
            self.sentences.append("\n")
        self.sentences.append("\n")

    def finalize_training(self):
        """
        Method used to finalize/start the training for the CoreNLP Module. Called before predictions can be made.
        :return:
        """
        train_text = ''.join(self.sentences)
        with codecs.open(self.train_path, "w+", "utf-8") as f:
            f.write(train_text)
        # Start the training script
        self.train()

    def train(self):
        """
        direct call to opennlp trainer with the input file. Writes the model to the output location
        :return:
        """
        subprocess.run(["opennlp", "SentenceDetectorTrainer", "-model", self.path, "-lang", "de",
                        "-data", self.train_path, "-encoding", "UTF-8"], stdout=sys.stdout, shell=True)


def prediction_from_sentences(sentences, special_characters=r"[\.,;:*#!?\"\(\{\}\[\]-_]\Z"):
    """
    Produces for a given sentence list the predictions.
    :param sentences: List of sentences
    :param special_characters:
    :return: Boolean values on sentence boundaries
    """
    prediction = []
    for sentence in sentences:
        # Might collide with some Annotations?
        sen = Tokenizer.tokenize_text(sentence)
        index_last_non_special = len(sen) - 1
        # TODO are sometimes special characters annotated as sentence end?
        for i, c in enumerate(reversed(sen)):
            if c not in special_characters and c != '\n':
                index_last_non_special -= i
                break
        for i, c in enumerate(sen):
            if i == index_last_non_special:
                prediction.append(True)
            else:
                prediction.append(False)
    return prediction


def calculateperformance(true_annotations, predicted_annotations):
    """
    Calculates common classification evaluation metrics.
    :param true_annotations: The ground truth annotations in a sequence. Values in [True, False]
    :param predicted_annotations: The predicted annotations in a sequence. Values in [True, False]
    :return: Dictionary with Sensitivity, Specificity, Precision, NPV, FPR, FDR, FNR, ACC, BACC, F1
    """
    total = len(true_annotations)
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    pos = 0
    neg = 0
    for true, pred in zip(true_annotations, predicted_annotations):
        if true:
            pos += 1
            if pred:
                true_pos += 1
            else:
                false_neg += 1
        else:
            neg += 1
            if pred:
                false_pos += 1
            else:
                true_neg += 1
    sens = true_pos/(true_pos+false_neg) if true_pos+false_neg > 0 else 0
    spec = true_neg/(false_pos+true_neg) if false_pos+true_neg > 0 else 0
    prec = true_pos/(true_pos+false_pos) if true_pos+false_pos > 0 else 0
    npv = true_neg/(true_neg+false_neg) if true_neg+false_neg > 0 else 0
    fpr = false_pos/(false_pos+true_neg) if false_pos+true_neg > 0 else 1
    fdr = false_pos/(false_pos+true_pos) if false_pos+true_pos > 0 else 1
    fnr = false_neg/(false_neg+true_pos) if false_neg+true_pos > 0 else 1
    f1 = 2*true_pos/(2*true_pos+false_pos+false_neg) if 2*true_pos+false_pos+false_neg > 0 else 0
    acc = (true_pos+true_neg)/total if total > 0 else 0
    bacc = (true_pos/pos + true_neg/neg)/2 if pos > 0 and neg > 0 else 0
    evaluation = {"sensitivity":    sens,
                  "specificity":    spec,
                  "precision":      prec,
                  "npv":            npv,
                  "fpr":            fpr,
                  "fdr":            fdr,
                  "fnr":            fnr,
                  "acc":            acc,
                  "bacc":           bacc,
                  "f1":             f1}
    return evaluation


def window_generator(token, windowsize):
    """
    Gives the generator to a sliding window. The token to look at is always in the middel of the window.
    That means whe have to refer to it from the back at index -windowsize/2
    :param token: The token list sliding over.
    :param windowsize: The maximum amount of tokens in one window.
    :return: Generator for a sliding window
    """
    if len(token) > windowsize/2:
        token_deque = deque(token)
        win_gen = deque()
        index = 0
        end_index = math.ceil(windowsize/2)
        # Window at the start of the text
        while index < end_index:
            win_gen.append(token_deque.popleft())
            index += 1
        yield win_gen
        # Window in the middle of the text
        while len(token_deque) > 0:
            if len(win_gen) >= windowsize:
                win_gen.popleft()
            win_gen.append(token_deque.popleft())
            yield win_gen
        # Window at the end of the text
        while len(win_gen) > math.ceil((windowsize + 1) / 2):
            win_gen.popleft()
            yield win_gen
    else:
        yield deque(token)
