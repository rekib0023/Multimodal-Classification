import pandas as pd
from config import config

def load_dataset(file_path, downsampling=False):
    _df = pd.read_json(config.INPUT_PATH + file_path, lines=True)

    if downsampling == True:
        count_class_0, count_class_1 = _df[config.LABEL].value_counts()

        df_class_0 = _df[_df[config.LABEL] == 0]
        df_class_1 = _df[_df[config.LABEL] == 1]

        df_class_0_under = df_class_0.sample(count_class_1)
        _df = pd.concat([df_class_0_under, df_class_1], axis=0)

    return _df