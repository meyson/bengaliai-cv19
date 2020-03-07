import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

if __name__ == '__main__':
    df = pd.read_csv('data/train.csv')
    print(df.head())
    df.loc[:, 'kfold'] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    X = df.image_id.values
    y = df[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values

    mskf = MultilabelStratifiedKFold(n_splits=5)

    for fold, (_trn, _val) in enumerate(mskf.split(X, y)):
        print('Train: ', _trn, 'VAL: ', _val)
        df.loc[_val, 'kfold'] = fold

    print('folds: ', df.kfold.value_counts())
    df.to_csv('data/train_folds.csv')
