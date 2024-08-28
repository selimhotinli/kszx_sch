import sys
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    p = subparsers.add_parser('download_act')
    p.add_argument('dr', type=int, help='Currently, only dr=5 is allowed')

    p = subparsers.add_parser('download_sdss')
    p.add_argument('survey', help='Survey name such as CMASS_North')
    
    args = parser.parse_args()

    if not hasattr(args, 'command'):
        parser.print_help()
        sys.exit(2)
    elif args.command == 'download_act':
        from . import act
        act.download(dr=args.dr)
    elif args.command == 'download_sdss':
        from . import sdss
        sdss.download(args.survey)
    else:
        parser.print_help()
        sys.exit(2)
