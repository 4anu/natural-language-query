from utils.loader import load_object, save_object
from utils.logger import Logger
import constants.main_constants as const

import pandas as pd
import os.path
import csv


class Glove:
    def __init__(self, file_name=const.GLOVE, load_if_exists=False):
        self.logger = Logger()
        self.logger.start_timer('Loading Glove embeddings..')
        if load_if_exists and os.path.exists(const.GLOVE_SAVE):
            self.words = load_object(const.GLOVE_SAVE)
        else:
            self.words = pd.read_table(file_name, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
            save_object(self.words, const.GLOVE_SAVE)
        self.logger.end_timer()

    def vector(self, word):
        return self.words.loc[word].as_matrix()
