# -*- coding: utf-8 -*-


from google.colab import files
uploaded = files.upload()

import pandas as pd
from io import BytesIO
from pathlib import Path
import pandas as pd  # version 1.0.3
from sklearn.metrics import brier_score_loss, roc_auc_score  # version 0.22.2
from xgboost import XGBClassifier  # version 1.0.2
import warnings
warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)
from ipywidgets import interact_manual, fixed, widgets  # 7.5.1
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, roc_auc_score  # version 0.22.2
from sklearn.model_selection import train_test_split, GridSearchCV  # version 0.22.2
from sklearn.calibration import CalibratedClassifierCV  # version 0.22.2
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import metrics


shots=pd.read_csv('shot.csv')
all_actions= pd.read_excel('all_actions.csv')


def shot_features_labels(shots):


  shot_features= shots.iloc[:, 1:6]
  shot_labels= shots.iloc[:, -1]


  df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(
    shot_features,
    shot_labels,
    test_size=0.10,
    random_state=42)
  return  df_X_train, df_X_test, df_y_train, df_y_test




def train_models():
  %%time
  modelss= [ XGBClassifier , LogisticRegression, RandomForestClassifier, svm.SVC ]
  for models in modelss:
    model= models()
    model.fit(X=df_X_train, y=df_y_train)
    score = model.score(df_X_test, df_y_test)
    print(score)



def xG_computation():

  all_action_X_test = all_actions.iloc[:, 1:10]

  probabilities = model.predict_proba(all_action_X_test)
  xG = probabilities[:, 1]
  dfs_predictions = pd.Series(xG, index=all_action_X_test.index)
  all_actions['xG']= dfs_predictions
  all_actions['reward']= all_actions['xG']* all_actions['shot_probabilities']

  return all_actions


