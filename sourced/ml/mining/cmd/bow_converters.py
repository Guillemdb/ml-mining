import argparse

from sourced.ml.core.models import BOW


def bow2vw(args: argparse.Namespace):
    bow = BOW().load(source=args.bow)
    bow.convert_bow_to_vw(args.output)
