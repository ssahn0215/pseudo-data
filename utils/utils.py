import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    argparser.add_argument(
        '-gpu', '--gpu-name',
        nargs='+',
        default=['0','1','2','3'],
        help="name of gpu to use")
    args = argparser.parse_args()
    return args
