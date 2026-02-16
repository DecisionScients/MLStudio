#%%
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

X, y = datasets.load_boston(return_X_y=True)
scaler = StandardScaler()    
X = scaler.fit_transform(X)
idx = np.arange(X.shape[0])
print(idx)
np.random.shuffle(idx)
print(idx)
X = np.array(X)
print(X[idx])
print(y[idx])

# %%
