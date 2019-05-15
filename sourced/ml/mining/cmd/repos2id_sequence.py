import logging
from uuid import uuid4

from sourced.ml.core.extractors import IdSequenceExtractor

from sourced.ml.mining.transformers.basic import create_uast_source, CsvSaver, Rower, \
    Uast2BagFeatures, UastDeserializer
from sourced.ml.mining.transformers.moder import Moder
from sourced.ml.mining.transformers.uast2bag_features import UastRow2Document
from sourced.ml.mining.utils.engine import pause, pipeline_graph


@pause
def repos2id_sequence(args):
    log = logging.getLogger("repos2id_distance")
    extractor = IdSequenceExtractor(args.split)
    session_name = "repos2roles_and_ids-%s" % uuid4()
    root, start_point = create_uast_source(args, session_name)
    if not args.skip_docname:
        mapper = Rower(lambda x: {"document": x[0][1],
                                  "identifiers": x[0][0]})
    else:
        mapper = Rower(lambda x: {"identifiers": x[0][0]})
    start_point \
        .link(Moder(args.mode)) \
        .link(UastRow2Document()) \
        .link(UastDeserializer()) \
        .link(Uast2BagFeatures(extractor)) \
        .link(mapper) \
        .link(CsvSaver(args.output)) \
        .execute()
    pipeline_graph(args, log, root)
