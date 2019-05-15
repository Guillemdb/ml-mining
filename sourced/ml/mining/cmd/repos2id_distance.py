import logging
from uuid import uuid4

from sourced.ml.core.extractors import IdentifierDistance

from sourced.ml.mining.transformers.basic import create_uast_source, CsvSaver, Rower, \
    Uast2BagFeatures, UastDeserializer
from sourced.ml.mining.transformers.moder import Moder
from sourced.ml.mining.transformers.uast2bag_features import UastRow2Document
from sourced.ml.mining.utils.engine import pause, pipeline_graph


@pause
def repos2id_distance(args):
    log = logging.getLogger("repos2roles_and_ids")
    extractor = IdentifierDistance(args.split, args.type, args.max_distance)
    session_name = "repos2roles_and_ids-%s" % uuid4()
    root, start_point = create_uast_source(args, session_name)

    start_point \
        .link(Moder(args.mode)) \
        .link(UastRow2Document()) \
        .link(UastDeserializer()) \
        .link(Uast2BagFeatures(extractor)) \
        .link(Rower(lambda x: {"identifier1": x[0][0][0],
                               "identifier2": x[0][0][1],
                               "distance": x[1]})) \
        .link(CsvSaver(args.output)) \
        .execute()
    pipeline_graph(args, log, root)
