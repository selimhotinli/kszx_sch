import sys
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    p = subparsers.add_parser('download_act')
    p.add_argument('dr', type=int, help='Currently, only dr=5 is allowed')

    p = subparsers.add_parser('download_sdss')
    p.add_argument('survey', help='Survey name such as CMASS_North')

    p = subparsers.add_parser('show')
    p.add_argument('filename')

    p = subparsers.add_parser('test')
    
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
    elif args.command == 'show':
        from . import io_utils
        io_utils.show_file(args.filename)
    elif args.command == 'test':
        from . import tests
        tests.run_all_tests()   # defined in kszx/tests/__init__.py
    else:
        parser.print_help()
        sys.exit(2)
