import os
import unittest

from sourced.ml.mining.tests import create_spark_for_test, tfidf_data
from sourced.ml.mining.transformers import BagFeatures2TermFreq


class Uast2TermFreqTests(unittest.TestCase):
    def setUp(self):
        self.sc = create_spark_for_test()
        self.bag2tf = BagFeatures2TermFreq()

    @unittest.skipIf(os.getenv("SKIP_SPARK_TESTS", True), "Skip ml_mining.bag2termfreq tests.")
    def test_call(self):
        tf = self.bag2tf(self.sc.sparkContext.parallelize(
            [((i["t"], i["d"]), i["v"]) for i in tfidf_data.dataset])) \
            .map(lambda r: {k[0]: v for k, v in r.asDict().items()}) \
            .collect()
        self.assertEqual({tfidf_data.readonly(i) for i in tf}, tfidf_data.term_freq_result)


if __name__ == "__main__":
    unittest.main()
