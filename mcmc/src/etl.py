import os
import numpy as np
import pandas as pd

def load_data(data_dir, data, **kwargs):
    path = os.path.join(data_dir, data)
    heart_data = pd.read_csv(path, header = None, delimiter = " ").values
    X = heart_data[:,:-1]
    X = np.hstack((np.ones((len(X),1)),X))  # add bias/offset term
    y = np.array([1 if i == 2 else 0 for i in heart_data[:,-1]]).reshape((-1,1))  # transform labels
    return X,y
