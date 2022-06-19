import sys
import data


def main(argv):
    df = data.load_data(path=argv[1])


if __name__ == '__main__':
    main(sys.argv)
