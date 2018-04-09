from types import SimpleNamespace


TOKENS = ['<UNK>', '<BEG>', '<END>', 'WHERE', 'AND', 'EQL', 'GT', 'LT']
COND = {
    'EQUAL': 'EQL',
    'GREATER_THAN': 'GT',
    'LESS_THAN': 'LT',
}
COND_OPERATORS = COND.values()
COND = SimpleNamespace(**COND)
