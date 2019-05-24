import argparse
import os
import sys
import tempfile
import unittest

from sourced.ml.core.extractors import IdentifiersBagExtractor

import sourced.ml.mining.tests.models as paths
from sourced.ml.mining.transformers.basic import (
    Counter, ParquetLoader, Uast2BagFeatures, UastDeserializer)
from sourced.ml.mining.transformers.moder import Moder
from sourced.ml.mining.transformers.uast2bag_features import UastRow2Document
from sourced.ml.mining.utils.docfreq import create_or_load_ordered_df
from sourced.ml.mining.utils.spark import create_spark


class DocumentFrequenciesUtilTests(unittest.TestCase):

    def test_load(self):
        args = argparse.Namespace(docfreq_in=paths.DOCFREQ, docfreq_out=None, min_docfreq=None,
                                  vocabulary_size=None)
        df_model = create_or_load_ordered_df(args, None, None)
        self.assertEqual(df_model.docs, 1000)

    @unittest.skipUnless(sys.version_info < (3, 7), "Python 3.7 is not yet supported")
    def test_create(self):
        session = create_spark("test_df_util")
        uast_extractor = ParquetLoader(session, paths.PARQUET_DIR) \
            .link(Moder("file")) \
            .link(UastRow2Document())
        ndocs = uast_extractor.link(Counter()).execute()
        uast_extractor = uast_extractor.link(UastDeserializer()) \
            .link(Uast2BagFeatures(IdentifiersBagExtractor()))
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, "df.asdf")
            args = argparse.Namespace(docfreq_in=None, docfreq_out=tmp_path, min_docfreq=1,
                                      vocabulary_size=1000)
            df_model = create_or_load_ordered_df(args, ndocs, uast_extractor)
            self.assertEqual(df_model.docs, ndocs)
            self.assertTrue(os.path.exists(tmp_path))

    @unittest.skipUnless(sys.version_info < (3, 7), "Python 3.7 is not yet supported")
    def test_error(self):
        with self.assertRaises(ValueError):
            create_or_load_ordered_df(argparse.Namespace(docfreq_in=None), 10, None)

        with self.assertRaises(ValueError):
            session = create_spark("test_df_util")
            uast_extractor = ParquetLoader(session, paths.PARQUET_DIR) \
                .link(Moder("file")) \
                .link(UastRow2Document()) \
                .link(UastDeserializer()) \
                .link(Uast2BagFeatures(IdentifiersBagExtractor()))
            create_or_load_ordered_df(argparse.Namespace(docfreq_in=None), None, uast_extractor)
