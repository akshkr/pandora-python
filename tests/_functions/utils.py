import re


def remove_symbols(text):
    return re.sub(r'^[a-zA-Z0-9]', '', text)