# -*- coding: utf-8 -*-
import pickle
import click
import logging
from pathlib import Path
import os.path

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

project_dir = Path(__file__).resolve().parents[2]

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--pca-var', default=0.95)
@click.option('--pca-output', type=click.Path(), default=os.path.join(project_dir, 'models/pca.p'))
@click.option('--colnames-output', type=click.Path(), default=os.path.join(project_dir, 'models/colnames.p'))
def main(input_filepath, output_filepath, pca_var, pca_output, colnames_output):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # read raw dataset
    logger.info('reading dataset at {}'.format(input_filepath))
    df = pd.read_csv(input_filepath)
    y = df['class']
    df = df.drop(columns=['class'])

    # convert to one-hot encoding
    logger.info('converting to one-hot encoded dataset')
    dum = pd.get_dummies(df)

    # save column names for later use
    logger.info('saving one-hot encoded column names to disk')
    pickle.dump(dum.columns, open(colnames_output, "wb"))

    # reduce dimensionality
    logger.info('reducing dimensionality with {} variance retained'.format(pca_var))
    pca = PCA(n_components=pca_var)
    trans = pca.fit_transform(dum)
    logger.info('resulting in a dataset with shape {}'.format(trans.shape))

    # save PCA model to cache location
    logger.info('saving PCA to disk')
    pickle.dump(pca, open(pca_output, "wb"))

    # create output dateframe
    logger.info('creating output dataframe')
    headers = ['feat_{}'.format(i) for i in range(trans.shape[1])]
    output_df = pd.DataFrame(data=trans, columns=headers)
    output_df = output_df.join(y.replace('e', 0).replace('p', 1))

    # write output
    output_df.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
