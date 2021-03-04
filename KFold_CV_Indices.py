import _pickle as cPickle
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import pandas as pd

# Open dataframe and convert it to array
dataframe = pd.read_pickle('nci_dados.pkl')

X = dataframe['image'].values
y = dataframe['label'].values

skf = KFold(n_splits=5, shuffle=True, random_state=42)

train_indices_list = []
test_indices_list = []
val_indices_list = []

for train_indices, test_indices in skf.split(X):
    
    X_train, X_test, y_train, y_test = train_test_split(X[train_indices], y[train_indices], test_size=0.20, random_state=42)
    
    t_indices = []
    v_indices = []

    for i, value in enumerate(X[train_indices]):
        if value in X_train:
            print("Value {} in train indices".format(i))
            t_indices.append(i)
        elif value in X_test:
            print("Value {} in val indices.".format(i))
            v_indices.append(i)
    
    print(np.shape(t_indices), np.shape(v_indices))

    train_indices_list.append(t_indices)
    val_indices_list.append(v_indices)
    test_indices_list.append(test_indices)

print(np.shape(train_indices_list), np.shape(val_indices_list), np.shape(test_indices_list))

with open('train_indices_list.pickle', 'wb') as t:
     cPickle.dump(train_indices_list, t, -1)

with open('test_indices_list.pickle', 'wb') as c:
     cPickle.dump(test_indices_list, c, -1)


with open('val_indices_list.pickle', 'wb') as d:
     cPickle.dump(val_indices_list, d, -1)

print('Train, Validation and Test Split Finished.')