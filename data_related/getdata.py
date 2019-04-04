'''
interface to read the data in hdf5 file
the default parameter is 'train', means load data from training data
or it will load the testing data
'''

import h5py
from sklearn.model_selection import train_test_split

def load(dataset='train'):
    if dataset == 'train':
        f = h5py.File("./data_related/mydata.h5")
        y = f['labels'].value
        x = f['data'].value
        
        f.close()
        x_train , x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=100)
        print(x_train.shape)
        return x_train, x_test, y_train, y_test
    
    
    elif dataset == 'test':
        f = h5py.File("./data_related/testdata_1.h5")
        x = f['data'].value
        y = f['labels'].value
        f.close()
        print(x.shape)
        return x, y
    
    else:
        raise NameError('The parameter can only be \'train\' or \'test\'')
