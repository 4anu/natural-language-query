from argparse import ArgumentParser

from models.data_model import DataModel
from models.embedding import Embedding
from models.glove import Glove

parser = ArgumentParser()
parser.add_argument('--gpu', action='store_true',
                    help='Use GPU')
parser.add_argument('--train_embedding', action='store_true',
                    help='Train word embedding for SQLNet(requires pre-trained model).')
parser.add_argument('--lr', default=0.02,
                    help='Learning rate')
parser.add_argument('--debug', action='store_true',
                    help='If set, use small data; used for fast debugging.')
parser.add_argument('-save', default='save',
                    help='Model save directory.')
args = parser.parse_args()

gl = Glove(load_if_exists=True)

dm = DataModel()
print(dm.__len__())

embedding = Embedding(glove=gl, gpu=args.gpu, train=args.train_embedding)
