from argparse import ArgumentParser

from services.common import load_data, make_token_to_index
from services.logger import Logger
from models.glove import Glove
from models.model import NLQModel
import constants.main_constants as const

parser = ArgumentParser()
parser.add_argument('--gpu', action='store_true',
                    help='Use GPU')
parser.add_argument('--train_embedding', action='store_false',
                    help='Train word embedding using pre-trained Glove.')
parser.add_argument('--size', default=256,
                    help='Model Size')
parser.add_argument('--layers', default=2,
                    help='Number of layers')
parser.add_argument('--lr', default=0.02,
                    help='Learning rate')
parser.add_argument('--ca', action='store_true',
                    help='Use column attention.')
parser.add_argument('--save', default='save',
                    help='Model save directory.')
parser.add_argument('--debug', action='store_true',
                    help='Fast debugging mode.')
parser.add_argument('--hard_reload', action='store_true',
                    help='All pre-processing will be done from scratch.')
args = parser.parse_args()
logger = Logger()

# TODO: Make a components check function in services.common to check for data, glove and directories.

logger.start_timer('Loading data..')
train_data, train_table_data, train_db = load_data(data_dir=const.DATA_DIR, split='train', debug=args.debug)
dev_data, dev_table_data, dev_db = load_data(data_dir=const.DATA_DIR, split='dev', debug=args.debug)
test_data, test_table_data, test_db = load_data(data_dir=const.DATA_DIR, split='test', debug=args.debug)
logger.end_timer()

glove = Glove(file_name=const.GLOVE, load_if_exists=(True and not args.hard_reload))
args.emb_size = glove.length

token_to_index = None
token_weights = None
if args.train_embedding:
    logger.start_timer('Making token dictionary..')
    token_to_index, token_weights = make_token_to_index(data=train_data, embedding=glove, use_extra_tokens=True,
                                                        load_if_exists=(True and not args.hard_reload))
    logger.end_timer()

nlq_model = NLQModel(embedding=glove, args=args, token_to_index=token_to_index, token_weights=token_weights)
