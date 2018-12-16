# -*- coding: utf-8 -*-
import click
import logging
import knn
import svm
from pathlib import Path
import os.path

models_dir = os.path.join(Path(__file__).resolve().parents[2], 'models')

@click.command()
@click.option('--use-cache', '-c', is_flag=True, default=False, help='use cached models for fast computation')
@click.option('--knn-filepath', '-k', default=os.path.join(models_dir, 'knn-high-res.p'), type=click.Path(), help='filepath for the knn cache file')
def main(use_cache, knn_filepath):
    """ Selects the best classification model. Optionally, cached models can be used
    by setting the --use-cache option.
    """
    logger = logging.getLogger(__name__)

    (knn_model, knn_score) = knn.optimize_knn(use_cache, knn_filepath)
    #(rf_model, rf_score) = rf.optimize_knn(use_cache, rf_filepath)
    #(svm_model, svm_score) = svm.optimize_svm(use_cache, svm_filepath)
    #(nn_model, nn_score) = nn.optimize_nn(use_cache, nn_filepath)


    # find optimal model


    print(knn_score)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
