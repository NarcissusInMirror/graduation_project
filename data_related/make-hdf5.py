from PIL import Image
import numpy as np
import csv
import random
import h5py

# list all the path or file needed 
IMAGE_DIR = './image/'
CSV_FILE = './annotation.csv'
SHUFFLIED_CSV_FILE = './shuffled_annotation.csv'
HDF5_FILE = './mydata.h5'
TEST_HDF5_FILE = './testdata.h5'

# initialize all the list needed
combine_list = []
train_list = []
test_list = []
train_image_list = []
train_label_list = []
test_image_list = []
test_label_list = []
        
# read the annotation file and bind the image name and the 
# label as tuple
with open(CSV_FILE, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        label = (row[1],row[2],row[3],row[4])
        combine_list.append((row[0] + '.png', label))
        
# shuffle the data
random.shuffle(combine_list)

sample_num = len(combine_list)
test_index = round(sample_num * 0.9)

# split the data into training and testing data
train_list = combine_list[0 : test_index]
test_list = combine_list[test_index : ]

# split the tuple
for i, group in enumerate(train_list):
    train_image_list.append(train_list[i][0])
    train_label_list.append(train_list[i][1])

for i, group in enumerate(test_list):
    test_image_list.append(test_list[i][0])
    test_label_list.append(test_list[i][1])

# initialize the numpy array needed
train_data = np.zeros((len(train_image_list), 3, 200, 200))
train_label = np.zeros((len(train_label_list), 4))
test_data = np.zeros((len(test_image_list), 3, 200, 200))
test_label = np.zeros((len(test_label_list), 4))

# write the data into the numpy array
for ii, _file in enumerate(train_image_list):
    img = Image.open(IMAGE_DIR + _file).resize((200, 200),Image.ANTIALIAS)
    img_data = np.array(img).transpose((2, 0, 1))
    train_data[ii, :, :, :] = img_data.astype(np.float32)
    train_label[ii, :] = np.array(train_label_list[ii]).astype(np.int)

for ii, _file in enumerate(test_image_list):
    img = Image.open(IMAGE_DIR + _file).resize((200, 200),Image.ANTIALIAS)
    img_data = np.array(img).transpose((2, 0, 1))
    test_data[ii, :, :, :] = img_data.astype(np.float32)
    test_label[ii, :] = np.array(test_label_list[ii]).astype(np.int)

    
print('train data shape:', train_data.shape)
print('train label shape:', train_label.shape)
print('test data shape:', test_data.shape)
print('train data shape:',test_label.shape)
print('writing file...')

# write the data into the hdf5 file
with h5py.File(HDF5_FILE, 'w') as f:
    f['data'] = train_data
    f['labels'] = train_label
    f.close()

with h5py.File(TEST_HDF5_FILE, 'w') as f:
    f['data'] = test_data
    f['labels'] = test_label
    f.close()

print('done...')

