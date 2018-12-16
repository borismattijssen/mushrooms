#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import click
import logging
import pickle
import os.path
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

data_dir = os.path.join(Path(__file__).resolve().parents[2], 'data')

@click.command()
@click.argument('input_filepath', default=os.path.join(data_dir, 'processed/mushrooms_pca.csv'), type=click.Path(exists=True))
@click.option('--first', default=0)
@click.option('--second', default=1)
def main(input_filepath, first, second):
    """ Scatter plot for two principal components.
    """
    logger = logging.getLogger(__name__)

    first_feat = 'feat_{}'.format(first)
    second_feat = 'feat_{}'.format(second)

    df = pd.read_csv(input_filepath)

    poisonous = df[df['class'] == 1]
    edible = df[df['class'] == 0]
    plt.scatter(poisonous[first_feat], poisonous[second_feat], label='p')
    plt.scatter(edible[first_feat], edible[second_feat], label='e')
    plt.legend(title='class')
    plt.xlabel('PC{}'.format(first))
    plt.ylabel('PC{}'.format(second))
    plt.title('Principal components scatter plot')
    plt.show()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
