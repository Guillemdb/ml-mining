import os
import sys
import tempfile
import unittest

from sourced.ml.core.extractors import ChildrenBagExtractor
from sourced.ml.core.models import QuantizationLevels

import sourced.ml.mining.tests.models as paths
from sourced.ml.mining.transformers.basic import ParquetLoader, UastDeserializer
from sourced.ml.mining.transformers.moder import Moder
from sourced.ml.mining.transformers.uast2bag_features import UastRow2Document
from sourced.ml.mining.utils.quant import create_or_apply_quant
from sourced.ml.mining.utils.spark import create_spark


class MyTestCase(unittest.TestCase):
    def test_apply(self):
        extractor = ChildrenBagExtractor()
        create_or_apply_quant(paths.QUANTLEVELS, [extractor])
        self.assertIsNotNone(extractor.levels)
        model_levels = QuantizationLevels().load(source=paths.QUANTLEVELS)._levels["children"]
        for key in model_levels:
            self.assertListEqual(list(model_levels[key]), list(extractor.levels[key]))

    @unittest.skipUnless(sys.version_info < (3, 7), "Python 3.7 is not yet supported")
    def test_create(self):
        session = create_spark("test_quant_util")
        extractor = ChildrenBagExtractor()
        with tempfile.NamedTemporaryFile(mode="r+b", suffix="-quant.asdf") as tmp:
            path = tmp.name
            uast_extractor = ParquetLoader(session, paths.PARQUET_DIR) \
                .link(Moder("file")) \
                .link(UastRow2Document()) \
                .link(UastDeserializer())
            create_or_apply_quant(path, [extractor], uast_extractor)
            self.assertIsNotNone(extractor.levels)
            self.assertTrue(os.path.exists(path))
            model_levels = QuantizationLevels().load(source=path)._levels["children"]
            for key in model_levels:
                self.assertListEqual(list(model_levels[key]), list(extractor.levels[key]))

    def test_error(self):
        with self.assertRaises(ValueError):
            create_or_apply_quant("", [], None)


if __name__ == "__main__":
    unittest.main()
