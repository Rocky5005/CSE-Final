import os
from ml_models import logistic_regression
from ml_models import naive_bayes
from ml_models import k_nearest_neighbors
from ml_models import support_vector
from ml_models import gradient_boost
from ml_models import random_forest
from support_functions import compare_performance_with_outliers
from support_functions import apply_rfe


filename = 'cleaned-framingham.csv'


def main():
    file_locate('main.py')
    logistic_regression.logistic_regression(filename)
    naive_bayes.naive_bayes(filename)
    k_nearest_neighbors.k_nearest(filename)
    gradient_boost(filename)
    random_forest(filename)
    compare_performance_with_outliers(filename)
    apply_rfe(filename)

def file_locate(__file__: str) -> None:
    pathstr = os.path.realpath(__file__)
    directory = os.path.dirname(pathstr)
    directory = directory.replace('src', 'data')
    os.chdir(directory)


if __name__ == '__main__':
    main()
