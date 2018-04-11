LOG_FILE = 'LogFile-{%}.log'
BATCH_SIZE = 64

# Glove and Embedding Constants
GLOVE_TOKENS = 6  # Billion
EMBEDDING_SIZE = 300  # Dimensions
GLOVE = '../glove/glove.{}B.{}d.txt'.format(GLOVE_TOKENS, EMBEDDING_SIZE)
GLOVE_SAVE = '../glove/loaded.pkl'
TOKEN_TO_IDX_SAVE = '../data/token_to_index.pkl'
TOKEN_WEIGHTS_SAVE = '../data/token_weights.pkl'
UNK_TOKEN = '<UNK>'
BEG_TOKEN = '<BEG'
END_TOKEN = '<END>'
UNK_IDX = 0
BEG_IDX = 1
END_IDX = 2

# Data Constants
DATA_DIR = '../data/'

# Debug Constants
DEBUG_DATA_SIZE = 1000
DEBUG_BATCH_SIZE = 16
