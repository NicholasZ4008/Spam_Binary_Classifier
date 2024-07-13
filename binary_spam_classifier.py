import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.sparse import issparse

import os
import mlflow
import mlflow.sklearn
import time
import string

RANDOM_STATE = 123


#IMPORT data:
DATASET_FILENAME = "spam.csv"
DATA_PATH = os.path.join("data", DATASET_FILENAME)

#encoding latin1 ensures removal of special characters. delimiter and usecols removes irrelevant columns and commas
df = pd.read_csv(DATA_PATH, encoding='latin1', delimiter=',', usecols=[0,1])
df.columns = ['label', 'message']  # Rename columns for clarity
print(df.head())

#PREPROCESS data:
X = df['message']
y = df['label'].apply(lambda x: 'spam' if 'spam' in x else 'not-spam')

#Config MLFLOW:
EXPERIMENT_NAME = "binary_classifier_spam"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

EXPERIMENT_ID = experiment.experiment_id
RUN_UID = str(int(time.time()))

print("Experiment ID:",EXPERIMENT_ID)
print("Experiment Name:", EXPERIMENT_NAME)

logreg_hyperparameters = {
    'penalty': 'l2',
    'solver': 'lbfgs',
    'random_state': RANDOM_STATE
}
nn_hyperparameters = {
    'learning_rate': 'adaptive',
    'alpha': 0.0001,
    'max_iter': 200,
    'activation': 'relu',
    'solver': 'adam',
    'random_state': RANDOM_STATE
}
rf_hyperparameters = {
    'max_depth': None,
    'n_estimators': 100, 
    'max_features': 'sqrt',
    'random_state': RANDOM_STATE
}

logreg = LogisticRegression(**logreg_hyperparameters)
nb = GaussianNB()
knn = KNeighborsClassifier(n_neighbors=5)
nn = MLPClassifier(**nn_hyperparameters)
rf = RandomForestClassifier(**rf_hyperparameters)

models = {
    "logreg": logreg, 
    # "nb": nb, 
    "knn": knn, 
    "nn": nn, 
    "rf": rf
}

