import pickle


def load_object(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def save_object(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
