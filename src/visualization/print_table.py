# -*- coding: utf-8 -*-
import click
import logging

import pandas as pd

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.option('--latex', '-l', is_flag=True, default=False, help='latex output')
def main(input_filepath, latex):
    """ Outputs the original csv as a table.
    """
    logger = logging.getLogger(__name__)

    df = pd.read_csv(input_filepath)
    out = df.head()
    if latex:
        out = out.to_latex()
    print(out)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
