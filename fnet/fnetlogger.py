import pandas as pd
import os

class FnetLogger(object):
    """Log values in a dict of lists."""
    def __init__(self, path_csv=None, columns=None):
        if path_csv is not None:
            df = pd.read_csv(path_csv)
            self.columns = list(df.columns)
            self.data = df.to_dict(orient='list')
        else:
            self.columns = columns
            self.data = {}
            for c in columns:
                self.data[c] = []

    def __repr__(self):
        return 'FnetLogger({})'.format(self.columns)

    def add(self, entry):
        if isinstance(entry, dict):
            for key, value in entry.items():
                self.data[key].append(value)
        else:
            assert len(entry) == len(self.columns)
            for i, value in enumerate(entry):
                self.data[self.columns[i]].append(value)

    def to_csv(self, path_csv):
        dirname = os.path.dirname(path_csv)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        pd.DataFrame(self.data)[self.columns].to_csv(path_csv, index=False)
