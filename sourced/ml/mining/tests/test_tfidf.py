import os
import unittest

from pyspark import Row
from sourced.ml.core.algorithms import log_tf_log_idf
from sourced.ml.core.models import DocumentFrequencies

from sourced.ml.mining.tests import create_spark_for_test, tfidf_data
from sourced.ml.mining.transformers.tfidf import TFIDF


@unittest.skipIf(os.getenv("SKIP_SPARK_TESTS", True), "Skip ml_mining.tfidf tests.")
class TFIDFTests(unittest.TestCase):
    def setUp(self):
        self.session = create_spark_for_test()

        df = DocumentFrequencies().construct(10, {str(i): i for i in range(1, 5)})
        self.docs = df.docs
        self.tfidf = TFIDF(df, df.docs, self.session.sparkContext)

        class Columns:
            """
            Stores column names for return value.
            """
            token = "t"
            document = "d"
            value = "v"

        self.tfidf.Columns = Columns

    def test_call(self):
        baseline = {
            Row(d=dict(i)["d"], t=dict(i)["t"],
                v=log_tf_log_idf(dict(i)["v"], int(dict(i)["t"]), self.docs))
            for i in tfidf_data.term_freq_result
        }

        result = self.tfidf(
            self.session.sparkContext
                .parallelize(tfidf_data.term_freq_result)
                .map(lambda x: Row(**dict(x)))).collect()
        self.assertEqual(set(result), baseline)


if __name__ == "__main__":
    unittest.main()
