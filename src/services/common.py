import json
import pickle

import constants.main_constants as const


def load_object(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def save_object(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_data(data_dir, split='train', debug=False):
    data_path = data_dir + split + '_tok.jsonl'
    table_path = data_dir + split + '_tok.tables.jsonl'
    db_path = data_dir + split + '.db'
    data = []
    table_data = {}
    with open(data_path) as f:
        for idx, line in enumerate(f):
            if debug and idx > const.DEBUG_DATA_SIZE:
                break
            data.append(json.loads(line.strip()))
    with open(table_path) as f:
        for _, line in enumerate(f):
            t_data = json.loads(line.strip())
            table_data[t_data['id']] = t_data
    return data, table_data, db_path


def make_token_to_index(data, use_extra_tokens=True):
    idx = 0
    token_to_index = dict()
    if use_extra_tokens:
        token_to_index[const.UNK_TOKEN] = const.UNK_IDX
        token_to_index[const.BEG_TOKEN] = const.BEG_IDX
        token_to_index[const.END_TOKEN] = const.END_IDX
        idx += 3

    for d in data:
        for token in d['question_tok']:
            if token not in token_to_index:
                token_to_index[token] = idx
                idx += 1
    save_object(token_to_index, const.TOKEN_TO_IDX_SAVE)
    return token_to_index
