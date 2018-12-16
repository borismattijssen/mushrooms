#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import click
import logging
import pickle
import os.path
from pathlib import Path

import matplotlib.pyplot as plt

models_dir = os.path.join(Path(__file__).resolve().parents[2], 'models')

@click.command()
@click.argument('input_filepath', default=os.path.join(models_dir, 'pca.p'), type=click.Path(exists=True))
def main(input_filepath):
    """ Plot for the variance explained of the PCA components.
    """
    logger = logging.getLogger(__name__)

    pca = pickle.load(open(input_filepath, "rb"))
    plt.plot(pca.explained_variance_)
    plt.xlabel('principal component')
    plt.ylabel('variance explained')
    plt.title('PCA variance explained')
    plt.show()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
