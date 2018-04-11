import numpy as np
import torch.utils.data as data_utils

import constants.main_constants as const


class DataModel(data_utils.DataLoader):
    def __init__(self, query_list, token_to_index, batch_size=const.BATCH_SIZE):
        query_len = [len(query) + 2 for query in query_list]  # BEG and END tokens included
        encoded_query_matrix = np.zeros([len(query_list), max(query_len)])
        for idx, query in enumerate(query_list):
            encoded_query = [const.BEG_IDX] + list(map(lambda x: token_to_index.get(x, const.UNK_IDX), query))\
                            + [const.END_IDX]
            encoded_query_matrix[idx, 0: len(encoded_query)] = encoded_query

        super(DataModel, self).__init__(encoded_query_matrix, batch_size)
