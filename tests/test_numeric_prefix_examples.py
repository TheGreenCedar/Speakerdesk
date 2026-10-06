"""Independent lexical examples, not acoustic or word-timestamp evidence."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'speakerdesk'))
from rolling_refinement import split_same_origin


class NumericPrefixExamples(unittest.TestCase):
    def test_single_cardinals_match_both_directions_and_preserve_raw_text(self):
        for digits, spoken in [('0', 'zero'), ('4', 'four'), ('11', 'eleven'),
                               ('18', 'eighteen'), ('20', 'twenty'),
                               ('70', 'seventy'), ('90', 'ninety')]:
            with self.subTest(digits=digits, direction='digits-to-words'):
                old = f'Please count {digits} pages.'
                new = f'Please count {spoken} pages. Keep the cover.'
                self.assertEqual(split_same_origin(old, new),
                                 (f'Please count {spoken} pages.', 'Keep the cover.'))
            with self.subTest(digits=digits, direction='words-to-digits'):
                old = f'Please count {spoken} pages.'
                new = f'Please count {digits} pages. Keep the cover.'
                self.assertEqual(split_same_origin(old, new),
                                 (f'Please count {digits} pages.', 'Keep the cover.'))

    def test_numeric_orthography_only_can_produce_a_verified_empty_suffix(self):
        for old, new in [('There are 12 pages.', 'There are twelve pages.'),
                         ('There are twelve pages.', 'There are 12 pages.')]:
            self.assertEqual(split_same_origin(old, new), (new, ''))

    def test_actual_appended_numeric_and_sentence_repetitions_survive(self):
        cases = [
            ('Count 4 pages.', 'Count four pages. Four pages.',
             ('Count four pages.', 'Four pages.')),
            ('Count four pages.', 'Count 4 pages. 4 pages.',
             ('Count 4 pages.', '4 pages.')),
            ('Count 11 pages.', 'Count eleven pages. Count eleven pages.',
             ('Count eleven pages.', 'Count eleven pages.')),
            ('Say 20.', 'Say twenty. Twenty.', ('Say twenty.', 'Twenty.')),
        ]
        for old, new, expected in cases:
            with self.subTest(old=old, new=new):
                self.assertEqual(split_same_origin(old, new), expected)

    def test_changed_values_never_match_even_at_the_last_lexical_token(self):
        for old, new in [('Count 4 pages.', 'Count five pages.'),
                         ('Count fourteen pages.', 'Count 4 pages.'),
                         ('I counted four.', 'I counted fourteen.'),
                         ('I counted three.', 'I counted thirteen.'),
                         ('I counted 18.', 'I counted nineteen.')]:
            with self.subTest(old=old, new=new):
                self.assertIsNone(split_same_origin(old, new))

    def test_leading_zero_identifiers_are_not_cardinal_equivalents(self):
        for old, new in [('Open file 004 now.', 'Open file four now.'),
                         ('Open file four now.', 'Open file 004 now.'),
                         ('The code is 00.', 'The code is zero.')]:
            with self.subTest(old=old, new=new):
                self.assertIsNone(split_same_origin(old, new))
        self.assertEqual(split_same_origin('Open file 004.', 'Open file 004. Blue.'),
                         ('Open file 004.', 'Blue.'))

    def test_decimals_and_ordinals_are_not_folded(self):
        for old, new in [('Use 4.5 liters.', 'Use four point five liters.'),
                         ('Use four point five liters.', 'Use 4.5 liters.'),
                         ('Use 4.0 liters.', 'Use four liters.'),
                         ('I counted 4.', 'I counted four point five.'),
                         ('I counted 4.', 'I counted four and a half.'),
                         ('Take the 4th copy.', 'Take the fourth copy.'),
                         ('I counted four.', 'I counted fourth.')]:
            with self.subTest(old=old, new=new):
                self.assertIsNone(split_same_origin(old, new))

    def test_unsupported_compounds_do_not_change_prefix_value(self):
        for old, new in [('Count 21 pages.', 'Count twenty one pages.'),
                         ('Count twenty-one pages.', 'Count 21 pages.'),
                         ('Count 100 pages.', 'Count one hundred pages.'),
                         ('I counted 20.', 'I counted twenty one.'),
                         ('I counted 20.', 'I counted twenty 1.'),
                         ('I counted twenty 1.', 'I counted 20 one.'),
                         ('I counted 3.', 'I counted three hundred.'),
                         ('I counted 3.', 'I counted three trillion.')]:
            with self.subTest(old=old, new=new):
                self.assertIsNone(split_same_origin(old, new))

    def test_dates_ranges_and_signed_forms_are_not_folded(self):
        for old, new in [('Meet on 4/7.', 'Meet on four/seven.'),
                         ('Take pages 4–7.', 'Take pages four–seven.'),
                         ('The reading is -4.', 'The reading is minus four.')]:
            with self.subTest(old=old, new=new):
                self.assertIsNone(split_same_origin(old, new))

    def test_fractional_compounds_never_fold_integer_prefixes(self):
        for digit, fraction in [('1', 'one half'), ('2', 'two thirds'),
                                ('4', 'four and a third'), ('4', 'four and two thirds'),
                                ('4', 'four and half'), ('6', 'six sevenths'),
                                ('1', 'one twentieth')]:
            with self.subTest(digit=digit, fraction=fraction):
                self.assertIsNone(split_same_origin(f'I counted {digit}.', f'I counted {fraction}.'))
                self.assertIsNone(split_same_origin(f'I counted {fraction}.', f'I counted {digit}.'))
        self.assertEqual(split_same_origin('I counted one half.', 'I counted one half. Blue.'),
                         ('I counted one half.', 'Blue.'))


if __name__ == '__main__':
    unittest.main()
