import unittest
import metric


class TestMetric(unittest.TestCase):

    def test_bleu_sentence_corpus_should_be_the_same(self):
        reference = "qalaye niazi is an ancient fortified area in paktia province in afghanistan"
        hypothesis = "qalaye niazi be a ancient fortified area in paktia province in afghanistan"
        score_sentence = metric.bleu_sentence(reference, hypothesis)
        score_corpus = metric.bleu_corpus([reference], [hypothesis])

        self.assertEquals(score_corpus, score_sentence)

    def test_accuracy_should_be_100_percent_if_ref_equals_translated(self):
        sentence = ["the dog walks on his 4 legs in the park"]

        self.assertEqual(1, metric.accuracy(sentence, sentence))

    def test_accuracy_should_penalize_extra_words(self):
        reference = ["the dog is walking there"]
        translation = ["the dog is walking there with his four hairy legs"]

        self.assertEqual(0.5, metric.accuracy(reference, translation))

    def test_accuracy_should_penalize_not_enough_words(self):
        reference = ["the dog walks on the beach by a sunny morning"]
        translation = ["the dog walks on the"]

        self.assertEqual(0.5, metric.accuracy(reference, translation))

    def test_accuracy_should_penalize_incorrect_translations(self):
        reference = ["the dog walks on his 4 legs in the park"]
        translation = ["the dog is on his 4 leg in the stadium"]

        self.assertEqual(0.7, metric.accuracy(reference, translation))

    def test_accuracy_should_examine_each_sentence(self):
        reference = ["the dog walks on his 4 legs in the park", "bob looked at the stars and saw jupiter"]
        translation = ["the dog is on his 4 leg in the stadium", "bob look at a star and saw some wood"]

        self.assertEqual(103/180, metric.accuracy(reference, translation))