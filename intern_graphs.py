import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
df = pd.read_csv('PS_20174392719_1491204439457_log.csv')

#line graph for all fraudulent transactions through out the month
df[df["isFraud"] == 1].groupby("step")["amount"].sum().plot()

#line graph for all transactions through out the month
df.groupby("step")["amount"].sum().plot()

#pie chart for all types
df_CI = df[df['type'] == "CASH_IN"]
df_CO = df[df['type'] == "CASH_OUT"]
df_D = df[df['type'] == "DEBIT"]
df_P = df[df['type'] == "PAYMENT"]
df_T = df[df['type'] == "TRANSFER"]
labels = ['CASH-IN', 'CASH-OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
sizes = [len(df_CI), len(df_CO), len(df_D), len(df_P), len(df_T)]
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%')
ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
ax.set_title('amount of transaction types')
plt.show()

#pie chart showing the fraudulent transactions vs flagged transactions
df_N = df[df['isFraud'] == 0]
df_F = df[df['isFraud'] == 1]
labels = ['LEGITIMATE', 'FRAUDULENT']
sizes = [len(df_N), len(df_F)]
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%')
ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
ax.set_title('legitimate vs fraudulent transactions')
plt.show()

#bar graph showing the amount of TRANSFER AND CASH_OUT transactions in fraudulent transactions
graphx = ['TRANSFER', 'CASH_OUT']
CTEs = [len(clean_fraud[clean_fraud['type'] == 0]),len(clean_fraud[clean_fraud['type'] == 1])]
x_pos = np.arange(len(graphx))
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, align='center', alpha=0.5)
ax.set_ylabel('number of times found in fraudulent data')
ax.set_xticks(x_pos)
ax.set_xticklabels(graphx)
ax.set_title('transaction type counts in fraudulent data')
ax.yaxis.grid(True)
plt.tight_layout()
plt.savefig('bar_plot.png')
plt.show()