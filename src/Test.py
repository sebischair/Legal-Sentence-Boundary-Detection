# Module used for testing the different modules

from Modules import *

import Tokenizer
import Splitter

import unittest


class TestSBDModules(unittest.TestCase):

    def __init__(self, stuff):
        unittest.TestCase.__init__(self, stuff)
        # Different test cases for all modules
        self.t_s1 = "Hello this should not be changed in any way."
        self.s_s1 = ["Hello this should not be changed in any way."]
        self.s_t1 = Tokenizer.tokenize_text(self.t_s1)
        self.t_s2 = "§1 Grundrechte\nDas ist das Grundrecht.\n§2 Keine Rechte\nNe das sind keine Rechte.\n"
        self.s_seg1 = ["§1 Grundrechte\nDas ist das Grundrecht.\n", "§2 Keine Rechte\nNe das sind keine Rechte.\n"]
        self.t_s3 = "§1 Grundrechte\n(1) Das ist das Grundrecht.\n(2) Das ist das zweite Grundrecht.\n" \
                    "§2 Keine Rechte\nNe das sind keine Rechte.\n"
        self.s_seg2 = ["§1 Grundrechte\n", "(1) Das ist das Grundrecht.\n",
                       "(2) Das ist das zweite Grundrecht.\n", "§2 Keine Rechte\nNe das sind keine Rechte.\n"]

    def setUp(self):
        self.rule = RuleModule()
        self.punkt = NLTKModule(fetch_module=True)
        self.temp = TemplateModule()
        self.crf = CRFModule()

    def test_rule(self):
        self.assertEqual(True, True, "Mockup test for rule module")
        pr_s1 = self.rule.predict_text(self.t_s1)
        p_s1 = Splitter.split_tokens(self.s_t1, pr_s1)
        self.assertEqual([self.s_t1], p_s1, "Template: Should not change the string with empty trigger list.")

    def test_punkt(self):
        self.assertEqual(True, True, "Mockup test for punkt module")
        pr_s1 = self.rule.predict_text(self.t_s1)
        p_s1 = Splitter.split_tokens(self.s_t1, pr_s1)
        self.assertEqual([self.s_t1], p_s1, "Template: Should not change the string with empty trigger list.")

    def test_template(self):
        self.assertEqual(True, True, "Mockup test for template module")
        p_s1 = self.temp.predict_text(self.t_s1)
        self.assertEqual(self.s_s1, p_s1, "Template: Should not change the string with empty trigger list.")
        self.temp.load_model()
        self.assertEqual([r"§[0-9]+", r"\([0-9]\)"], self.temp.trigger_list)
        p_s2 = self.temp.predict_text(self.t_s2)
        self.assertEqual(self.s_seg1, p_s2, "Template: The first law segment was not segmented correct.")
        p_s3 = self.temp.predict_text(self.t_s3)
        self.assertEqual(self.s_seg2, p_s3, "Template: The second law segment was not segmented correct.")

    @unittest.skip("CRF: Is not implemented yet")
    def test_crf(self):
        pass

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
