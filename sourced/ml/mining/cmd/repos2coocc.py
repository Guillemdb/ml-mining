import logging
from uuid import uuid4

from sourced.ml.core.extractors import IdentifiersBagExtractor

from sourced.ml.mining.transformers.basic import Cacher, Counter, create_uast_source,\
    Repartitioner, Uast2BagFeatures, UastDeserializer
from sourced.ml.mining.transformers.coocc import CooccConstructor, CooccModelSaver
from sourced.ml.mining.transformers.moder import Moder
from sourced.ml.mining.transformers.uast2bag_features import UastRow2Document
from sourced.ml.mining.utils.docfreq import create_or_load_ordered_df
from sourced.ml.mining.utils.engine import pause, pipeline_graph


@pause
def repos2coocc(args):
    log = logging.getLogger("repos2coocc")
    id_extractor = IdentifiersBagExtractor(docfreq_threshold=args.min_docfreq,
                                           split_stem=args.split)
    session_name = "repos2coocc-%s" % uuid4()
    root, start_point = create_uast_source(args, session_name)

    uast_extractor = start_point \
        .link(Moder("file")) \
        .link(UastRow2Document()) \
        .link(Repartitioner.maybe(args.partitions, args.shuffle)) \
        .link(Cacher.maybe(args.persist))
    log.info("Extracting UASTs...")
    ndocs = uast_extractor.link(Counter()).execute()
    log.info("Number of documents: %d", ndocs)
    uast_extractor = uast_extractor.link(UastDeserializer())

    df_model = create_or_load_ordered_df(
        args, ndocs, uast_extractor.link(Uast2BagFeatures(id_extractor)))

    token2index = root.session.sparkContext.broadcast(df_model.order)
    uast_extractor \
        .link(CooccConstructor(token2index=token2index,
                               token_parser=id_extractor.id2bag.token_parser,
                               namespace=id_extractor.NAMESPACE)) \
        .link(CooccModelSaver(args.output, df_model)) \
        .execute()
    pipeline_graph(args, log, root)
