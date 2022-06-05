# Module to tokenizer legal documents
# If executed as standalone script, it will output the tokenized text in the same path

import sys
import getopt
import codecs
import json
import re
from collections import deque

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
    elif input_directory.endswith(".txt"):
        with codecs.open(input_directory, 'r', 'utf-8') as f:
            text = f.read()
        token = tokenize_text(text)
    else:
        raise ValueError("No possible input file. Use .json or .txt")
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


def tokenize_text(text):
    """
    Tokenizes a string.
    :param text: String
    :return: Tokens
    """
    token = []
    running_word = ""
    for c in text:
        if re.match(alphanumeric, c):
            running_word += c
        else:
            if running_word != "":
                token.append(running_word)
            if c not in filter_character:
                token.append(c)
            running_word = ""
    if running_word != "":
        token.append(running_word)
    return token


def tokenize_text_with_special(text):
    """
    Tokenizes a string. Does not filter any characters.
    :param text: The String to be tokenized.
    :return: Tokens
    """
    token = []
    running_word = ""
    for c in text:
        if re.match(alphanumeric, c):
            running_word += c
        else:
            if running_word != "":
                token.append(running_word)
            token.append(c)
            running_word = ""
    if running_word != "":
        token.append(running_word)
    return token


if __name__ == "__main__":
    argv = sys.argv[1:]
    i_d = ''
    o_d = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["indir=","outdir="])
    except getopt.GetoptError:
        print("Tokenizer.py -i <input_directory> -o <output_directory>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("Tokenizer.py -i <input_directory> -o <output_directory>")
        elif opt in ("-i", "--indir"):
            i_d = arg
        elif opt in ("-o", "--outdir"):
            o_d = arg
    tokenize(i_d, o_d)

