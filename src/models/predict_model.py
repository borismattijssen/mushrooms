#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import click
import pickle
import logging
import knn
from pathlib import Path
import os.path
import pandas as pd

models_dir = os.path.join(Path(__file__).resolve().parents[2], 'models')

def default_none(ctx, param, value):
    if len(value) == 0:
        return [
            os.path.join(models_dir, 'knn-high-res.p'),
            os.path.join(models_dir, 'rf-grid-search.p')
        ]

@click.command()
@click.argument('input_filepath', nargs=1)
@click.argument('models', nargs=-1, callback=default_none)
def main(input_filepath, models):
    """ INPUT_FILEPATH A CSV containing mushrooms.\n
    MODELS a list of classification models.
    """
    logger = logging.getLogger(__name__)

    # load model
    model = find_best_model(logger, models)

    # load pca model and one-hot column names
    pca = pickle.load(open(os.path.join(models_dir, 'pca.p'), "rb"))
    colnames = pickle.load(open(os.path.join(models_dir, 'colnames.p'), "rb"))

    # load data
    df = pd.read_csv(input_filepath)
    dum = pd.get_dummies(df).reindex(columns = colnames, fill_value=0)
    pcad = pca.transform(dum)

    print("\n".join([str(x) for x in model.predict(pcad)]))


    # find optimal model

def find_best_model(logger, models):
    estimators = []
    scores = []

    # load all models
    for model_path in models:
        try:
            (e, s) = load(model_path)
            estimators.append(e)
            scores.append(s)
            logger.info('Added model {}'.format(model_path))
        except Exception as e:
            logger.info('Failed to add model {}'.format(model_path))

    # find the index of the highest score
    ix = scores.index(max(scores))

    # return the estimator with the highest score
    return estimators[ix]

def load(model_file):
    """ Load a model from a filepath.
    """

    # if we should load from cache and the cache file exists
    clf = pickle.load(open(model_file, "rb"))
    return (clf.best_estimator_, clf.best_score_)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
