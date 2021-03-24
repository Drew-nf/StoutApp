import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance, to_graphviz
df = pd.read_csv('PS_20174392719_1491204439457_log.csv')

#cleaning data to only useful values
clean = df.loc[(df.type == 'TRANSFER') | (df.type == 'CASH_OUT')]
randomState = 5
np.random.seed(randomState)
clean_isF = clean['isFraud']
del clean['isFraud']
# Eliminate columns shown to be irrelevant for analysis in the EDA
clean = clean.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis = 1)
# Binary-encoding of labelled data in 'type'
clean.loc[clean.type == 'TRANSFER', 'type'] = 0
clean.loc[clean.type == 'CASH_OUT', 'type'] = 1
clean.type = clean.type.astype(int) # convert dtype('O') to dtype(int)


clean_fraud = clean.loc[clean_isF == 1]
clean_nonFraud = clean.loc[clean_isF == 0]
print('\nThe percent of transactions when the oldbalanceDest & newbalanceDest are equal to 0 even though the transacted amount is non-zero is: {}'.\
format(len(clean_fraud.loc[(clean_fraud.oldbalanceDest == 0) & \
(clean_fraud.newbalanceDest == 0) & (clean_fraud.amount)]) / (1.0 * len(clean_fraud))))
print('\nThe fraction of genuine transactions when the oldbalanceDest and newbalanceDest are equal to 0 even though the transacted amount is non-zero is: {}'.\
format(len(clean_nonFraud.loc[(clean_nonFraud.oldbalanceDest == 0) & \
(clean_nonFraud.newbalanceDest == 0) & (clean_nonFraud.amount)]) / (1.0 * len(clean_nonFraud))))
    
#creating new columns to to find errors in balances
clean['errorBalanceOrig'] = clean.newbalanceOrig + clean.amount - clean.oldbalanceOrg
clean['errorBalanceDest'] =clean.oldbalanceDest + clean.amount - clean.newbalanceDest

#training set
trainclean, testclean, trainclean_isF, testclean_isF = train_test_split(clean, clean_isF, test_size = 0.2, \
                                                random_state = randomState)
#find the accuracy of ML algorithm
weights = (clean_isF == 0).sum() / (1.0 * (clean_isF == 1).sum())
clf = XGBClassifier(max_depth = 3, scale_pos_weight = weights, \
                n_jobs = 4)
probabilities = clf.fit(trainclean, trainclean_isF).predict_proba(testclean)
print('AUPRC = {}'.format(average_precision_score(testclean_isF, \
                                              probabilities[:, 1])))
