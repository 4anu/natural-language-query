import numpy as np
import torch.utils.data as data_utils

import constants.main_constants as const


class DataModel(data_utils.DataLoader):
    def __init__(self, batch_size=const.BATCH_SIZE):
        dataset = np.random.rand(1000)
        super(DataModel, self).__init__(dataset, batch_size)

# TODO: Make data model
