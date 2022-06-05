import Tokenizer

import re

special = r"[\.,;:*#!?\"\(\)\{\}\[\]-_]\Z"
filtered_token = [' ', '\t', '\v', '\r', '\f', '\n']
filtered_not_newline = [' ', '\t', '\v', '\r', '\f']
alphanumeric = Tokenizer.alphanumeric


def split_tokens(token_list, sentence_boundary, filter_newline=True):
    """
    This method splits up a given token list based on the predicted sentence boundaries. The output is still tokenized
    in the same way. The tokens are filtered based on the filtered_token list.
    :param token_list: List of the tokenized text to segment into sentences
    :param sentence_boundary: List of truth values for each token (stating the end of a sentence)
    :param filter_newline: Boolean whether newlines should be filtered from the token list
    :return: List of tokenized sentences.
    """
    # Create the necessary list structures for the collection of sentences.
    # The boolean setence_end is needed to track when parsing the tokens, if there was already a sentence end.
    # This is needed because the last non-special token (e.g. normal word) is annotated as the sentence end,
    # all following special characters need to be appended to the sentence (such as dots)
    sentence_list = []
    running_sentence = []
    sentence_end = False
    if filter_newline:
        token_filter = filtered_token
    else:
        token_filter = filtered_not_newline
    for token, boundary in zip(token_list, sentence_boundary):
        if token in token_filter:
            continue
        if not boundary and not sentence_end:
            running_sentence.append(token)
        elif boundary:
            running_sentence.append(token)
            sentence_end = True
        elif sentence_end:
            if re.match(special, token):
                running_sentence.append(token)
            elif not filter_newline and token == "\n":
                running_sentence.append(token)
                sentence_list.append(running_sentence)
                running_sentence = []
                sentence_end = False
            else:
                sentence_list.append(running_sentence)
                running_sentence = [token]
                sentence_end = False
    if len(running_sentence) > 0:
        sentence_list.append(running_sentence)
    return sentence_list


def split_sentences(text, sentence_boundary):
    """
    This method splits up a given text into sentences based on the predicted sentence boundaries. The output is not
    tokenized. The tokens are filtered based on the filtered_token list. Spaces are not removed.
    :param text: A string to segment into sentences
    :param sentence_boundary:
    :return: A list of non-tokenized sentences.
    """
    # Create the necessary list structures for the collection of sentences.
    # The boolean setence_end is needed to track when parsing the tokens, if there was already a sentence end.
    # This is needed because the last non-special token (e.g. normal word) is annotated as the sentence end,
    # all following special characters need to be appended to the sentence (such as dots)
    # Important: Newline counts as a token, because it is used in the predictions!
    sentence_list = []
    running_sentence = []
    sentence_end = False
    last_special = False
    last_alphanum = False
    last_newline = False
    index = 0
    for c in text:
        if index < len(sentence_boundary) and (sentence_boundary[index] or sentence_end):
            if re.match(alphanumeric, c):
                # index stays the same, still same word
                running_sentence.append(c)
                sentence_end = True
                last_special = False
                last_alphanum = True
                last_newline = False
            elif c in filtered_token:
                if not last_newline:
                    index += 1
                if c == "\n":
                    # There is no sentence break on newline
                    index += 1
                    last_newline = True
                else:
                    last_newline = False
                sentence_end = False
                last_special = False
                last_alphanum = False
                sentence_list.append(''.join(running_sentence))
                running_sentence = []
            # Otherwise the character is some special token not filtered out
            else:
                running_sentence.append(c)
                if last_alphanum or last_special:
                    index += 1
                sentence_end = True
                last_special = True
                last_alphanum = False
                last_newline = False
        else:
            if re.match(alphanumeric, c):
                # index stays the same, still same word
                running_sentence.append(c)
                if last_special:
                    index += 1
                last_special = False
                last_alphanum = True
                last_newline = False
            elif c in filtered_token:
                if not last_newline:
                    index += 1
                if c == "\n":
                    # There is no sentence break on newline
                    index += 1
                    running_sentence.append(" ")
                    last_newline = True
                elif c == " " and not last_newline:
                    running_sentence.append(c)
                    last_newline = False
                else:
                    last_newline = False
                last_special = False
                last_alphanum = False
            # Otherwise the character is some special token not filtered out
            else:
                running_sentence.append(c)
                if last_alphanum or last_special:
                    index += 1
                last_special = True
                last_alphanum = False
                last_newline = False
    if len(running_sentence) > 0 and running_sentence[0] != " ":
        sentence_list.append(''.join(running_sentence))
    return sentence_list
