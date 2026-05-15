from argparse import ArgumentParser
from pathlib import Path

from spectrum import read_spectrum, plot_spectrum

###############################################################################


def setup_argparse():
    """
    Sets up argument parser
    ----------
    args:
        None
    ----------
    returns:
        parser              argparse.ArgumentParser                 argument parser object
    """
    parser = ArgumentParser(description="")
    parser.add_argument('-f', '--frame',     action="store", type=int, required=True, help="Index of frame to analyse")
    parser.add_argument('-s', '--spec_path', action="store", type=str, required=True, help="Path to spectrum data file")
    return parser


def parse_args(parser):
    """
    Parses arguments and verifies them for validity
    ----------
    args:
        parser              argparse.ArgumentParser                 argument parser object
    ----------
    returns:
        args                argparse.Namespace                      namespace object of parsed args
    """
    args = parser.parse_args()

    args.spec_path = Path(args.spec_path)
    if not args.spec_path.is_file():
        print("The specified spectrum file does not exist")
        return

    if args.frame < 0:
        print("Frame number must be positive or zero")
        return

    return args                                             # if arguments were parsed successfully, return them

###############################################################################


if __name__ == "__main__":
    args = parse_args(setup_argparse())                     # parse arguments
    if args is not None:                                    # proceed only if parsing was successful
        spectrum = read_spectrum(args.spec_path)
        plot_spectrum(spectrum, args.frame)
