import pandas as pd
from conf.conf import logging


def get_data(link: str) -> pd.DataFrame:
    """ 
    Getting table from csv 
    """
    logging.info('Extracting dataset')
    df = pd.read_csv(link)
    logging.info('Dataset is extracted')
    
    return df
