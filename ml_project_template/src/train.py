import pandas as pd
from sklearn import preprocessing

Training_Data= None
FOLD= None

FOLD_MAPPING={
    0:[1,2,3,4],
    1:[0,2,3,4],
    2:[0,1,3,4],
    3:[0,1,2,4],
    4:[0,1,2,3]
}

if __name__="__main__":
    df=pd.read_csv(Training_Data)
    train_df=df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df=df(df.kfold==FOLD)

    ytrain=train_df.target.values
    yvalid=valid_df.target.values

    train_df=train_df.drop(["id","target","kfold"],axis=1)
    valid_df=valid_df.drop(["id","target","kfold"],axis=1)
