import logging
from uuid import uuid4

from sourced.ml.core.extractors import RoleIdsExtractor

from sourced.ml.mining.transformers.basic import create_uast_source, CsvSaver,\
    Rower, UastDeserializer
from sourced.ml.mining.transformers.moder import Moder
from sourced.ml.mining.transformers.uast2bag_features import Uast2Features, UastRow2Document
from sourced.ml.mining.utils.engine import pause, pipeline_graph


@pause
def repos2roles_and_ids(args):
    log = logging.getLogger("repos2roles_and_ids")
    session_name = "repos2roles_and_ids-%s" % uuid4()
    extractor = RoleIdsExtractor()
    root, start_point = create_uast_source(args, session_name)

    start_point \
        .link(Moder("file")) \
        .link(UastRow2Document()) \
        .link(UastDeserializer()) \
        .link(Uast2Features(extractor)) \
        .link(Rower(lambda x: {"identifier": x["roleids"][0], "role": x["roleids"][1]})) \
        .link(CsvSaver(args.output)) \
        .execute()
    pipeline_graph(args, log, root)
