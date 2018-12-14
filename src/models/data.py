from pathlib import Path
import os.path
import pandas as pd

def load_pca_data():
    """ Load the PCA'd mushroom dataset.
    """

    # find path
    project_dir = Path(__file__).resolve().parents[2]
    filepath = os.path.join(project_dir, 'data/processed/mushrooms_pca.csv')

    # load dataset
    df = pd.read_csv(filepath)

    # split features and label
    y = df['class']
    X = df.drop(columns=['class']).values
    
    return (X,y)
