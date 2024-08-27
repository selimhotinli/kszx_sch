import sys
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    p = subparsers.add_parser('download_act')
    p.add_argument('dr', type=int, help='Currently, only dr=5 is allowed')
    
    # More commands to be added later...
    
    args = parser.parse_args()

    if not hasattr(args, 'command'):
        parser.print_help()
        sys.exit(2)
    elif args.command == 'download_act':
        from . import act
        act.download(dr=args.dr)
    else:
        parser.print_help()
        sys.exit(2)

