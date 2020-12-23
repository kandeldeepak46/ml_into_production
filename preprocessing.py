import os
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OneHotEncoder

# Load data and save indices of columns
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)

DATA_FILE = os.path.join(DATA_DIR, 'data.csv')
if not os.path.isfile(DATA_FILE):
    raise FileNotFoundError("please check the files and directory")

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')


if not os.path.isdir(MODEL_DIR):
    os.mkdir(MODEL_DIR)


with open(DATA_FILE, 'r') as file:
    df = pd.read_csv(file)

features = df.drop("left", 1).columns
pickle.dump(features, open(f"{MODEL_DIR}/features.pickle", "wb"))

# Fit and save an OneHotEncoder
columns_to_fit = ["sales", "salary"]
enc = OneHotEncoder(sparse=False).fit(df.loc[:, columns_to_fit])
pickle.dump(enc, open(f"{MODEL_DIR}/encoder.pickle", "wb"))

# Transform variables, merge with existing df and keep column names
column_names = enc.get_feature_names(columns_to_fit)
encoded_variables = pd.DataFrame(
    enc.transform(df.loc[:, columns_to_fit]), columns=column_names
)
df = df.drop(columns_to_fit, 1)
df = pd.concat([df, encoded_variables], axis=1)

# Fit and save model
X, y = df.drop("left", 1), df.loc[:, "left"]
clf = LGBMClassifier().fit(X, y)
pickle.dump(clf, open(f"{MODEL_DIR}/model.pickle", "wb"))


if __name__ == '__main__':
    main()
