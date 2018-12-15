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

@click.command()
@click.argument('input_filepath', nargs=1)
@click.argument('model', nargs=1, default=os.path.join(models_dir, 'rf-grid-search.p'))
def main(input_filepath, model):
    """ INPUT_FILEPATH A CSV containing mushrooms. The file should contain a CSV header.\n
    MODEL a filepath to a pickled classification model.
    """
    logger = logging.getLogger(__name__)

    # load model
    model = pickle.load(open(model, "rb"))

    # load pca model and one-hot column names
    pca = pickle.load(open(os.path.join(models_dir, 'pca.p'), "rb"))
    colnames = pickle.load(open(os.path.join(models_dir, 'colnames.p'), "rb"))

    # load data
    df = pd.read_csv(input_filepath)
    dum = pd.get_dummies(df).reindex(columns = colnames, fill_value=0)
    pcad = pca.transform(dum)

    classes = [str(x).replace("1", "p").replace("0", "e") for x in model.predict(pcad)]
    print("\n".join(classes))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
