import sys
import unittest

from pyspark.sql.functions import lit

from sourced.ml.mining.tests.models import PARQUET_DIR, SIVA_DIR
from sourced.ml.mining.transformers.basic import (
    Collector, HeadFiles, Ignition, LanguageExtractor, LanguageSelector)
from sourced.ml.mining.transformers.basic import create_engine, create_parquet_loader


class LoaderTest(unittest.TestCase):
    @unittest.skipUnless(sys.version_info < (3, 7), "Python 3.7 is not yet supported")
    def test_parquet(self):
        languages1 = ["Python", "Java"]
        languages2 = ["Java"]

        engine = create_engine("test", SIVA_DIR)
        res = Ignition(engine) \
            .link(HeadFiles()) \
            .link(LanguageExtractor()) \
            .link(LanguageSelector(languages1)) \
            .link(Collector()) \
            .execute()
        self.assertEqual({x.lang for x in res}, set(languages1))

        res = Ignition(engine) \
            .link(HeadFiles()) \
            .link(LanguageExtractor()) \
            .link(LanguageSelector(languages2)) \
            .link(Collector()) \
            .execute()
        self.assertEqual({x.lang for x in res}, set(languages2))

        res = Ignition(engine) \
            .link(HeadFiles()) \
            .link(LanguageExtractor()) \
            .link(LanguageSelector(languages2, blacklist=True)) \
            .link(Collector()) \
            .execute()
        self.assertEqual(set(), {x.lang for x in res} & set(languages2))

        res = Ignition(engine) \
            .link(HeadFiles()) \
            .link(LanguageExtractor()) \
            .link(LanguageSelector([])) \
            .link(Collector()) \
            .execute()
        self.assertEqual(set(), {x.lang for x in res})

        parquet_loader = create_parquet_loader("test_parquet", repositories=PARQUET_DIR)
        df = parquet_loader.execute()
        with self.assertRaises(AttributeError):
            LanguageSelector(languages1)(df)

        df_with_lang = df.withColumn("lang", lit("BestLang"))
        self.assertEqual(0, len(LanguageSelector(languages1)(df_with_lang).collect()))

        self.assertEqual(df_with_lang.collect(),
                         LanguageSelector(["BestLang"])(df_with_lang).collect())


if __name__ == "__main__":
    unittest.main()
