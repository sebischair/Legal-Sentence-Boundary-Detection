class FeatureExtractor:

    def __init__(self, scope_before=0, scope_after=0, on_token=True):
        self.scope_before = scope_before
        self.scope_after = scope_after
        self.on_token = on_token
        self.name = "bias"

    def process_feature(self, token_list, previous_features):
        for i, token in enumerate(token_list):
            # before features
            start_index = i - self.scope_before if i-self.scope_before >= 0 else 0
            for j in range(start_index, i):
                feature_name = self.name + '[' + str(j-i) + ']'
                if len(previous_features) > i:
                    value = self.process_token_feature(token_list[j])
                    if value is not None:
                        previous_features[i][feature_name] = value
                else:
                    value = self.process_token_feature(token_list[j])
                    if value is not None:
                        previous_features.append({feature_name: value})
            end_index = i + self.scope_after + 1 if i + self.scope_after + 1 <= len(token_list) else len(token_list)
            for j in range(i+1, end_index):
                feature_name = self.name + '[' + str(j-i) + ']'
                if len(previous_features) > i:
                    value = self.process_token_feature(token_list[j])
                    if value is not None:
                        previous_features[i][feature_name] = value
                else:
                    value = self.process_token_feature(token_list[j])
                    if value is not None:
                        previous_features.append({feature_name: value})
            feature_name = self.name + '[' + str(0) + ']'
            if len(previous_features) > i:
                value = self.process_token_feature(token_list[i])
                if value is not None:
                    previous_features[i][feature_name] = value
            else:
                value = self.process_token_feature(token_list[i])
                if value is not None:
                    previous_features.append({feature_name: value})
        return previous_features

    def process_token_feature(self, token):
        return 1


class TitleExtractor(FeatureExtractor):

    def __init__(self, scope_before=0, scope_after=0, on_token=True):
        FeatureExtractor.__init__(self, scope_before, scope_after, on_token)
        self.name = "title"

    def process_feature(self, token_list, previous_features):
        for i, token in enumerate(token_list):
            # before features
            start_index = i - self.scope_before if i-self.scope_before >= 0 else 0
            for j in range(start_index, i):
                feature_name = self.name + '[' + str(j-i) + ']'
                if token_list[j].isupper():
                    value = True
                else:
                    value = False
                if len(previous_features) > i:
                    if value is not None:
                        previous_features[i][feature_name] = value
                else:
                    if value is not None:
                        previous_features.append({feature_name: value})
            end_index = i + self.scope_after + 1 if i + self.scope_after + 1 <= len(token_list) else len(token_list)
            for j in range(i+1, end_index):
                feature_name = self.name + '[' + str(j-i) + ']'
                if len(previous_features) > i:
                    value = False
                    if value is not None:
                        previous_features[i][feature_name] = value
                else:
                    value = False
                    if value is not None:
                        previous_features.append({feature_name: value})
            feature_name = self.name + '[' + str(0) + ']'
            if len(previous_features) > i:
                value = False
                if value is not None:
                    previous_features[i][feature_name] = value
            else:
                value = False
                if value is not None:
                    previous_features.append({feature_name: value})
        return previous_features


class TruthExtractor(FeatureExtractor):

    def __init__(self, scope_before=0, scope_after=0, on_token=True):
        FeatureExtractor.__init__(self, scope_before, scope_after, on_token)
        self.name = "t"


class CombinationExtractor(FeatureExtractor):

    def __init__(self, extractor1, extractor2):
        FeatureExtractor.__init__(self, scope_before=max(extractor1.scope_before, extractor2.scope_before),
                                  scope_after=max(extractor1.scope_after, extractor2.scope_after))
        self.name = "c" + extractor1.name + extractor2.name
        self.extractor1 = extractor1
        self.extractor2 = extractor2
        if isinstance(extractor1, TruthExtractor):
            self.process_truth = True
        elif isinstance(extractor2, TruthExtractor):
            self.process_truth = True
            self.extractor1, self.extractor2 = self.extractor2, self.extractor1
        else:
            self.process_truth = False

    def process_token_feature(self, token):
        if self.process_truth:
            if self.extractor1.process_token_feature(token):
                return self.extractor2.process_token_feature(token)
            else:
                return None
        else:
            concat = str(self.extractor1.process_token_feature(token)) + \
                     str(self.extractor2.process_token_feature(token))
            return concat


class LengthExtractor(FeatureExtractor):

    def __init__(self, scope_before=0, scope_after=0, on_token=True):
        FeatureExtractor.__init__(self, scope_before, scope_after, on_token)
        self.name = "len"

    def process_token_feature(self, token):
        return len(token)


class LastNewlineExtractor(FeatureExtractor):

    def __init__(self, short_step=3, medium_step=7):
        FeatureExtractor.__init__(self, 0, 0, True)
        self.short_step = short_step
        self.medium_step = medium_step
        self.name = "LastNew"
        self.last_newline = 1

    def process_token_feature(self, token):
        duration = self.last_newline
        if token == "\n":
            duration = 0
        else:
            self.last_newline += 1
        if duration < self.short_step:
            return "Short"
        elif duration < self.medium_step:
            return "Medium"
        else:
            return "Long"


class SignatureExtractor(FeatureExtractor):

    def __init__(self, scope_before=0, scope_after=0, on_token=True):
        FeatureExtractor.__init__(self, scope_before, scope_after, on_token)
        self.name = "sig"

    def process_token_feature(self, token):
        sig = []
        for c in token:
            if c.isnumeric():
                sig.append("N")
            elif c.islower():
                sig.append("c")
            elif c.isupper():
                sig.append("C")
            else:
                sig.append("S")
        return ''.join(sig)


class LowercaseExtractor(FeatureExtractor):

    def __init__(self, scope_before=0, scope_after=0, on_token=True):
        FeatureExtractor.__init__(self, scope_before, scope_after, on_token)
        self.name = "lower"

    def process_token_feature(self, token):
        return token.lower()


class WordsPerLineExtractor(FeatureExtractor):

    def __init__(self):
        FeatureExtractor.__init__(self, 0, 0, True)
        self.name = "WLine"


class NumberExtractor(TruthExtractor):

    def __init__(self, scope_before=0, scope_after=0, on_token=True):
        TruthExtractor.__init__(self, scope_before, scope_after, on_token)
        self.name = "isnum"

    def process_token_feature(self, token):
        return True if token.isnumeric() else False


class SpecialExtractor(FeatureExtractor):

    def __init__(self, scope_before=0, scope_after=0, on_token=True):
        FeatureExtractor.__init__(self, scope_before, scope_after, on_token)
        self.name = "spec"

    def process_token_feature(self, token):
        if len(token) > 1 or token.isalnum():
            return "No"
        # TODO might want to deferentiate between dot and others
        elif token in [".", "!", "?", ";"]:
            return "End"
        elif token in ["(", "[", "{"]:
            return "Open"
        elif token in [")", "]", "}"]:
            return "Close"
        elif token in ["\r", "\n"]:
            return "Newline"
        else:
            return "S"


class UpperExtractor(TruthExtractor):

    def __init__(self, scope_before=0, scope_after=0, on_token=True):
        TruthExtractor.__init__(self, scope_before, scope_after, on_token)
        self.name = "isup"

    def process_token_feature(self, token):
        return True if token.isupper() else False


class LowerExtractor(TruthExtractor):

    def __init__(self, scope_before=0, scope_after=0, on_token=True):
        TruthExtractor.__init__(self, scope_before, scope_after, on_token)
        self.name = "islow"

    def process_token_feature(self, token):
        return True if token.islower() else False


class SBDExtractor(TruthExtractor):

    def __init__(self, scope_before=0, scope_after=0, on_token=True):
        TruthExtractor.__init__(self, scope_before, scope_after, on_token)
        self.name = "specSB"

    def process_token_feature(self, token):
        if token in [".", "!", "?", ";"]:
            return True
        else:
            return False


class ParenthesesExtractor(TruthExtractor):

    def __init__(self, scope_before=0, scope_after=0, on_token=True):
        TruthExtractor.__init__(self, scope_before, scope_after, on_token)
        self.name = "specPar"

    def process_token_feature(self, token):
        if token in ["(", "[", "{", ")", "]", "}"]:
            return True
        else:
            return False


class NewlineExtractor(TruthExtractor):

    def __init__(self, scope_before=0, scope_after=0, on_token=True):
        TruthExtractor.__init__(self, scope_before, scope_after, on_token)
        self.name = "specPar"

    def process_token_feature(self, token):
        if token in ["\r", "\n"]:
            return True
        else:
            return False


class SBDNewlineExtractor(TruthExtractor):

    def __init__(self):
        TruthExtractor.__init__(self, 0, 0, True)
        self.name = "break"
        self.sentence_boundary = False
        self.newline = False

    def process_token_feature(self, token):
        if token in [".", "!", "?"]:
            self.sentence_boundary = True
            self.newline = False
        elif token == '\n':
            if self.sentence_boundary:
                self.newline = True 
            else:
                self.newline = False
        else:
            self.newline = False
            self.sentence_boundary = False
        if self.newline:
            return True
        else:
            return False


class AbreviationExtractor(TruthExtractor):

    def __init__(self, scope_before=0, scope_after=0, on_token=True, abbreviation_list=None):
        TruthExtractor.__init__(self, scope_before, scope_after, on_token)
        if abbreviation_list is None:
            self.abbreviations = ["Abs", "usw", "bzw", "Bl", "ff", "Nr", "ca", "Hr"]
        else:
            self.abbreviations = abbreviation_list
        self.name = "Abr"

    def process_token_feature(self, token):
        if token in self.abbreviations:
            return True
        else:
            return False
