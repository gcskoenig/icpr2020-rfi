name='confounding2'

'''
Fitting a model
'''
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
import math

N=10**5
dataset = np.loadtxt('data/{}.csv'.format(name), dtype=np.float32)
D = np.arange(1, 4)

splitpoint = math.floor(N*0.9)
ix_train = np.arange(0, splitpoint, 1)
ix_test = np.arange(splitpoint, N, 1)

X_train, y_train = dataset[ix_train, :-1], dataset[ix_train,-1]
X_test, y_test = dataset[ix_test, :-1], dataset[ix_test,-1]

# Linear Model
print('Linear Model')

model = LinearRegression()
model.fit(X_train[:, D], y_train)
names = ['c', 'x1', 'x2', 'x3', 'y']
print(names[1:-1])
print(model.coef_)
 
# ['x1', 'x2', 'x3']
# [0.99642247 1.1678281  0.66819274]

y_pred = model.predict(X_test[:, D])
risk = mean_squared_error(y_test, y_pred)
print(risk)


r = permutation_importance(model, X_test[:, D], y_test, n_repeats=30, random_state=0)
print(names[1:-1])
print(r['importances_mean'])
# ['x1', 'x2', 'x3']
# [0.30951335 0.86202417 0.17510705]



'''
Computing RFI
'''

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from rfi import rfi, cfi, plot_rfis, create_2nd_order_knockoff, paired_t
import pandas as pd
import copy

names = np.array(names)

D = np.arange(1, 4)

loss = lambda x, y : np.power(x-y, 2)

rfis = []
pvals = []
rfinames = []
Dnames = [r'$X_1$', r'$X_2$', r'$X_3$']

G = np.array([])
res = rfi(model.predict, loss, G, X_train, X_test, y_test, D, n_repeats=30)
rfis.append([res[0], res[1]])
pvals.append([res[2], paired_t(res[2], res[3])])
rfinames.append(r'$PFI$')

res = cfi(model.predict, loss, X_train, X_test, y_test, D, n_repeats=30)
rfis.append([res[0], res[1]])
pvals.append([res[2], paired_t(res[2], res[3])])
rfinames.append(r'$CFI$')

G = np.array([0])
res = rfi(model.predict, loss, G, X_train, X_test, y_test, D, n_repeats=30)
rfis.append([res[0], res[1]])
pvals.append([res[2], paired_t(res[2], res[3])])
rfinames.append(r'$RFI_j^{C}$')

# G = np.array([0,1])
# res = rfi(model.predict, mean_squared_error, G, X_train, X_test, y_test, D, n_repeats=30)
# rfis.append([res[0], res[1]])
# rfinames.append( r"$RFI_j^{C,X_1}$")

# G = np.array([0,3])
# res = rfi(model.predict, mean_squared_error, G, X_train, X_test, y_test, D, n_repeats=30)
# rfis.append([res[0], res[1]])
# rfinames.append( r"$RFI_j^{C,X_3}$")

# G = np.array([0,2])
# res = rfi(model.predict, mean_squared_error, G, X_train, X_test, y_test, D, n_repeats=30)
# rfis.append([res[0], res[1]])
# rfinames.append(r"$RFI_j^{C,X_2}$")

# G = np.array([0,3])
# res = rfi(model.predict, mean_squared_error, G, X_train, X_test, y_test, D, n_repeats=30)
# rfis.append([res[0], res[1]])
# rfinames.append(r"$RFI_j^{C,X_3}$")

# G = np.array([3])
# res = rfi(model.predict, mean_squared_error, G, X_train, X_test, y_test, D, n_repeats=30)
# rfis.append([res[0], res[1]])
# rfinames.append(r"$RFI_j^{X_3}$")

# G = np.array([3])
# mean_rfi_2, std_rfi_2, rfi_2 = rfi(model.predict, mean_squared_error, G, X_train, X_test, y_test, D, n_repeats=30)
# print(mean_rfi_2)

pvals = np.array(pvals)
print(pvals[:, 1, 0, :])
print(pvals[:, 1, 0, :] > 0.01)
np.save('results/{}_pvals.npy'.format(name), pvals)

# [[0.         0.         0.        ]
#  [0.         0.         0.        ]
#  [0.         0.         0.03283623]]
# [[False False False]
#  [False False False]
#  [False False  True]]

plot_rfis(rfis, Dnames, rfinames, 'results/{}.pdf'.format(name), figsize=(20, 10))

