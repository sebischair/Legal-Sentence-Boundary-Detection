from CRFTrainer import *
from FeatureExtractors import *


class FeatureGenerator:

    def __init__(self, feature_list=[[FeatureExtractor()]]):
        self.feature_list = feature_list

    def generate_feature_combinations(self):
        length = LengthExtractor(7, 7)
        sign = SignatureExtractor(5, 5)
        low = LowercaseExtractor(7, 7)
        islow = LowerExtractor(3, 3)
        isup = UpperExtractor(3, 3)
        num = NumberExtractor(3, 3)
        spec = SpecialExtractor(10, 10)
        features = [length, sign, low, islow, isup, num, spec]
        yield features


if __name__ == "__main__":
    feature_exploration("../data", "Models/Testing", "xml", FeatureGenerator(), log_file="test_new_jug.log", runs=3)
