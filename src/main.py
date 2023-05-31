import os
from ml_models import logistic_regression
from mlmodels import naive_bayes
from mlmodels import k_nearest
from mlmodels import support_vector
from mlmodels import gradient_boost
from mlmodels import random_forest
from mlmodels import compare_performance_with_outliers
from mlmodels import apply_rfe


filename = 'cleaned-framingham.csv'


def main():
    file_locate('main.py')
    logistic_regression.logistic_regression(filename)
    naive_bayes(filename)
    k_nearest(filename)
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
