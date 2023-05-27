import os
from mlmodels import logistic_regression


filename = 'cleaned-framingham.csv'


def main():
    file_locate()
    logistic_regression(filename)


def file_locate() -> None:
    __file__ = 'main.py'
    pathstr = os.path.realpath(__file__)
    directory = os.path.dirname(pathstr)
    directory = directory.replace('src', 'data')
    os.chdir(directory)


if __name__ == '__main__':
    main()
