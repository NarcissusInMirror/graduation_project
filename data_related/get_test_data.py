import h5py
from sklearn.model_selection import train_test_split

def load_test():
    f = h5py.File("testdata.h5")
    x_test = f['data'].value
    y_test = f['labels'].value
    return x_test, y_test
